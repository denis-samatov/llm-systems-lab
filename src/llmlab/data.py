"""Corpus loading, tokenizer training and deterministic batch sampling.

Corpus: RuMedTop3 (github.com/sb-ai-lab/MedBench, Apache-2.0 code; records derived from
RuMedPrime, CC BY 3.0). Raw files are verified against committed SHA-256 sums before use.

Sampling contract (what makes single-device, DDP and FSDP2 runs comparable): the global
batch of optimizer step ``s`` is a pure function of ``(seed, s)``. Every micro-batch on every
rank is a fixed slice of that global batch, so a run with ``world_size=1, grad_accum=W*A``
sees exactly the same samples per optimizer step as ``world_size=W, grad_accum=A``.
Resuming at step ``s`` therefore needs no data-loader state beyond the step counter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

SPLITS = ("train", "dev", "test")
EOS = "<|endoftext|>"
PAD = "<|pad|>"
RUMED_URL = "https://raw.githubusercontent.com/sb-ai-lab/MedBench/main/data/RuMedTop3/{split}_v1.jsonl"


@dataclass(frozen=True)
class Record:
    idx: str
    text: str
    code: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_sums(raw_dir: Path) -> dict[str, str]:
    sums = {}
    for line in (raw_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*").removeprefix("./")] = digest
    return sums


def load_rumed(split: str, raw_dir: str | Path = "data/raw") -> list[Record]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    raw_dir = Path(raw_dir)
    path = raw_dir / f"{split}_v1.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `llmlab data download`")
    expected = _expected_sums(raw_dir).get(path.name)
    if expected is None or sha256_file(path) != expected:
        raise ValueError(f"{path}: checksum mismatch, refusing to train on drifted data")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.append(Record(row["idx"], row["symptoms"].strip(), row["code"].strip()))
    return out


def download_rumed(raw_dir: str | Path = "data/raw") -> dict[str, str]:
    """Download into a staging name, verify, then move: never leaves a half-written file."""
    import urllib.request

    raw_dir = Path(raw_dir)
    sums = _expected_sums(raw_dir)
    for split in SPLITS:
        target = raw_dir / f"{split}_v1.jsonl"
        if target.exists() and sha256_file(target) == sums[target.name]:
            continue
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(RUMED_URL.format(split=split), tmp)
        if sha256_file(tmp) != sums[target.name]:
            tmp.unlink()
            raise ValueError(f"downloaded {split} split does not match SHA256SUMS")
        tmp.replace(target)
    return sums


def train_tokenizer(texts: list[str], vocab_size: int, path: str | Path):
    """Byte-level BPE trained on the *train split only* (no dev/test leakage)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def encode_stream(tok, texts: list[str]) -> torch.Tensor:
    """Concatenate documents separated by EOS into one int32 token stream."""
    eos = tok.token_to_id(EOS)
    ids: list[int] = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eos)
    return torch.tensor(ids, dtype=torch.int32)


class TokenStreams:
    """Train/dev/test token streams plus the tokenizer, built once and cached on disk."""

    def __init__(self, raw_dir: str | Path, tokenizer_path: str | Path, vocab_size: int):
        tok_path = Path(tokenizer_path)
        train = load_rumed("train", raw_dir)
        if not tok_path.exists():
            train_tokenizer([r.text for r in train], vocab_size, tok_path)
        self.tokenizer = load_tokenizer(tok_path)
        if self.tokenizer.get_vocab_size() > vocab_size:
            raise ValueError("tokenizer vocabulary larger than model vocab_size")
        self.tokenizer_sha256 = sha256_file(tok_path)
        self.streams = {"train": encode_stream(self.tokenizer, [r.text for r in train])}
        for split in ("dev", "test"):
            self.streams[split] = encode_stream(self.tokenizer, [r.text for r in load_rumed(split, raw_dir)])


class BatchSampler:
    """Deterministic, resumable sampler of fixed-length windows from a token stream."""

    def __init__(self, stream: torch.Tensor, seq_len: int, seed: int):
        if stream.numel() <= seq_len + 1:
            raise ValueError("token stream shorter than one sequence")
        self.stream = stream.long()
        self.seq_len = seq_len
        self.seed = seed

    def global_offsets(self, step: int, global_batch: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        return torch.randint(0, self.stream.numel() - self.seq_len - 1, (global_batch,), generator=g)

    def micro_batch(
        self, step: int, micro_idx: int, rank: int, world: int, micro_bs: int, grad_accum: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs, targets) for one micro-batch of one rank.

        Layout of the global batch: slot ``(micro_idx * world + rank)`` of size ``micro_bs``.
        """
        if not (0 <= micro_idx < grad_accum and 0 <= rank < world):
            raise ValueError("micro_idx/rank out of range")
        offsets = self.global_offsets(step, micro_bs * grad_accum * world)
        slot = micro_idx * world + rank
        mine = offsets[slot * micro_bs : (slot + 1) * micro_bs]
        idx = mine[:, None] + torch.arange(self.seq_len + 1)[None, :]
        window = self.stream[idx]
        return window[:, :-1].contiguous(), window[:, 1:].contiguous()

    def eval_batches(self, n_batches: int, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Fixed evaluation windows (independent of training step)."""
        out = []
        for b in range(n_batches):
            g = torch.Generator().manual_seed(10_000_019 + b)
            offs = torch.randint(0, self.stream.numel() - self.seq_len - 1, (batch_size,), generator=g)
            idx = offs[:, None] + torch.arange(self.seq_len + 1)[None, :]
            w = self.stream[idx]
            out.append((w[:, :-1].contiguous(), w[:, 1:].contiguous()))
        return out
