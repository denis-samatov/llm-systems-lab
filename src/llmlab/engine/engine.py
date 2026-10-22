"""A small LLM serving engine: continuous batching over a paged KV cache.

Scheduling follows the token-budget design of vLLM V1: a request is a list of tokens with a
``num_computed_tokens`` cursor; every step the scheduler hands out up to
``max_num_batched_tokens`` new tokens, first to running requests (decode = 1 token, or the
next chunk of a long prompt), then to waiting requests (prefill, possibly chunked). Prefill
and decode tokens of different requests share one forward pass. When the KV pool runs out,
the most recently admitted running request is *preempted* (blocks freed, recomputed later).

Baselines in the same module, for benchmarks and correctness tests:

* :func:`generate_no_cache` - recompute the whole sequence each token (O(n^2) work).
* :func:`generate_static_batch` - contiguous KV cache, fixed batches, left padding; the
  batch runs until its longest request finishes (HF ``generate``-style static batching).
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field

import torch

from llmlab.engine.attention import ContiguousAttention, PagedAttention, build_paged_batch
from llmlab.engine.kv_cache import BlockAllocator, ContiguousKVCache, NoFreeBlocks, PagedKVCache, block_hash
from llmlab.model import GPT


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 32
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 1.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    req_id: str
    prompt_ids: list[int]
    params: SamplingParams
    arrival: float = 0.0
    output_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    block_hashes: list[bytes] = field(default_factory=list)
    status: Status = Status.WAITING
    cached_prompt_tokens: int = 0
    num_preemptions: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None
    finish_reason: str | None = None
    generator: torch.Generator | None = None

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None) -> int:
    if params.temperature <= 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits.float() / params.temperature, dim=-1)
    if params.top_p < 1.0:
        sorted_p, idx = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_p, 0) - sorted_p < params.top_p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(0, idx, sorted_p)
        probs /= probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


@dataclass(frozen=True)
class EngineConfig:
    num_blocks: int = 512
    block_size: int = 16
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 512
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = True


@dataclass
class StepStats:
    num_seqs: int
    num_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    blocks_used: int
    seconds: float


class LLMEngine:
    def __init__(self, model: GPT, cfg: EngineConfig, kv_dtype: torch.dtype = torch.float32):
        mc = model.cfg
        self.model = model.eval()
        self.cfg = cfg
        self.allocator = BlockAllocator(cfg.num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.cache = PagedKVCache(
            mc.n_layers,
            cfg.num_blocks,
            cfg.block_size,
            mc.n_kv_heads or mc.n_heads,
            mc.d_model // mc.n_heads,
            dtype=kv_dtype,
        )
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.step_stats: list[StepStats] = []
        self.clock = time.perf_counter

    # ---------------------------------------------------------------- public API
    def add_request(self, req: Request) -> None:
        if req.num_tokens + req.params.max_new_tokens > self.model.cfg.max_seq_len:
            raise ValueError(f"{req.req_id}: prompt + max_new_tokens exceeds max_seq_len")
        if (
            math.ceil((req.num_tokens + req.params.max_new_tokens) / self.cfg.block_size)
            > self.cfg.num_blocks
        ):
            raise ValueError(f"{req.req_id}: request can never fit in the KV cache")
        if req.params.seed is not None:
            req.generator = torch.Generator().manual_seed(req.params.seed)
        self.waiting.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.inference_mode()
    def step(self) -> list[Request]:
        t0 = self.clock()
        scheduled = self._schedule()
        if not scheduled:
            return []
        n_prefill = sum(n for r, n in scheduled if r.num_computed_tokens < len(r.prompt_ids))
        batch = build_paged_batch(
            [r.block_table for r, _ in scheduled],
            [r.num_computed_tokens for r, _ in scheduled],
            [n for _, n in scheduled],
            self.cfg.block_size,
        )
        tokens, positions, sample_rows, offset = [], [], [], 0
        for r, n in scheduled:
            ids = r.all_ids
            tokens.extend(ids[r.num_computed_tokens : r.num_computed_tokens + n])
            positions.extend(range(r.num_computed_tokens, r.num_computed_tokens + n))
            offset += n
            if r.num_computed_tokens + n == r.num_tokens:  # last known token computed -> sample
                sample_rows.append(offset - 1)
        logits = self.model(
            torch.tensor([tokens]),
            torch.tensor([positions]),
            PagedAttention(self.cache, batch),
            logits_index=torch.tensor(sample_rows, dtype=torch.long),
        )
        done: list[Request] = []
        row = 0
        now = self.clock()
        for r, n in scheduled:
            r.num_computed_tokens += n
            self._register_full_blocks(r)
            if r.num_computed_tokens == r.num_tokens:
                tok = sample(logits[row], r.params, r.generator)
                row += 1
                r.output_ids.append(tok)
                if r.first_token_time is None:
                    r.first_token_time = now
                if tok in r.params.stop_token_ids:
                    self._finish(r, "stop", now)
                    done.append(r)
                elif len(r.output_ids) >= r.params.max_new_tokens:
                    self._finish(r, "length", now)
                    done.append(r)
        self.step_stats.append(
            StepStats(
                len(scheduled),
                len(tokens),
                n_prefill,
                len(tokens) - n_prefill,
                self.allocator.num_used(),
                self.clock() - t0,
            )
        )
        return done

    def run(self) -> list[Request]:
        while self.has_unfinished():
            before = len(self.step_stats)
            self.step()
            if len(self.step_stats) == before:
                # Nothing could be scheduled although work remains: the pool is too small or
                # blocks leaked. Fail loudly instead of spinning forever.
                raise NoFreeBlocks(
                    f"scheduler made no progress: {len(self.waiting)} waiting, {len(self.running)} running, "
                    f"{self.allocator.num_free()}/{self.cfg.num_blocks} blocks free"
                )
        return self.finished

    # ---------------------------------------------------------------- scheduling
    def _blocks_needed(self, r: Request, n_new: int) -> int:
        return max(0, math.ceil((r.num_computed_tokens + n_new) / self.cfg.block_size) - len(r.block_table))

    def _allocate(self, r: Request, count: int) -> bool:
        if self.allocator.num_free() < count:
            return False
        for _ in range(count):
            r.block_table.append(self.allocator.allocate())
        return True

    def _schedule(self) -> list[tuple[Request, int]]:
        budget = self.cfg.max_num_batched_tokens
        scheduled: list[tuple[Request, int]] = []
        # 1) running requests, FCFS. Preempt from the back when blocks run out.
        i = 0
        while i < len(self.running) and budget > 0:
            r = self.running[i]
            n = min(r.num_tokens - r.num_computed_tokens, budget)
            while not self._allocate(r, self._blocks_needed(r, n)):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is r:
                    break
            else:
                scheduled.append((r, n))
                budget -= n
                i += 1
                continue
            break  # r itself was preempted
        # 2) waiting requests while there is budget and sequence capacity.
        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            r = self.waiting[0]
            if r.num_computed_tokens == 0 and self.cfg.enable_prefix_caching:
                self._match_prefix(r)
            remaining = r.num_tokens - r.num_computed_tokens
            n = min(remaining, budget)
            if n < remaining and not self.cfg.enable_chunked_prefill:
                self._unmatch_prefix(r)
                break
            if not self._allocate(r, self._blocks_needed(r, n)):
                self._unmatch_prefix(r)
                break
            self.waiting.pop(0)
            r.status = Status.RUNNING
            self.running.append(r)
            scheduled.append((r, n))
            budget -= n
        return scheduled

    def _match_prefix(self, r: Request) -> None:
        """Reuse cached blocks for the longest fully-cached prefix (keeping >= 1 token to compute)."""
        bs = self.cfg.block_size
        ids = r.all_ids
        max_full = (len(ids) - 1) // bs
        parent = None
        for i in range(max_full):
            h = block_hash(parent, ids[i * bs : (i + 1) * bs])
            b = self.allocator.lookup(h)
            if b is None:
                break
            r.block_table.append(b)
            r.block_hashes.append(h)
            parent = h
        r.num_computed_tokens = len(r.block_table) * bs
        r.cached_prompt_tokens = max(r.cached_prompt_tokens, r.num_computed_tokens)

    def _unmatch_prefix(self, r: Request) -> None:
        if r.num_computed_tokens == 0 or r.status is not Status.WAITING:
            return
        for b in r.block_table:
            self.allocator.release(b)
        r.block_table.clear()
        r.block_hashes.clear()
        r.num_computed_tokens = 0

    def _register_full_blocks(self, r: Request) -> None:
        if not self.cfg.enable_prefix_caching:
            return
        bs = self.cfg.block_size
        ids = r.all_ids
        while len(r.block_hashes) < r.num_computed_tokens // bs:
            i = len(r.block_hashes)
            h = block_hash(r.block_hashes[-1] if r.block_hashes else None, ids[i * bs : (i + 1) * bs])
            r.block_hashes.append(h)
            self.allocator.register(r.block_table[i], h)

    def _free_blocks(self, r: Request) -> None:
        for b in reversed(r.block_table):  # release tail first: LRU evicts suffix blocks first
            self.allocator.release(b)
        r.block_table.clear()
        r.block_hashes.clear()

    def _preempt(self, r: Request) -> None:
        self.running.remove(r)
        self._free_blocks(r)
        r.num_computed_tokens = 0
        r.num_preemptions += 1
        r.status = Status.WAITING
        self.waiting.insert(0, r)

    def _finish(self, r: Request, reason: str, now: float) -> None:
        self.running.remove(r)
        self._free_blocks(r)
        r.status = Status.FINISHED
        r.finish_reason = reason
        r.finish_time = now
        self.finished.append(r)


# -------------------------------------------------------------------- baselines
@torch.inference_mode()
def generate_no_cache(model: GPT, prompt: list[int], params: SamplingParams) -> list[int]:
    gen = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
    ids = list(prompt)
    out: list[int] = []
    for _ in range(params.max_new_tokens):
        logits = model(torch.tensor([ids]))[0, -1]
        tok = sample(logits, params, gen)
        out.append(tok)
        ids.append(tok)
        if tok in params.stop_token_ids:
            break
    return out


@torch.inference_mode()
def generate_static_batch(
    model: GPT,
    prompts: list[list[int]],
    params: list[SamplingParams],
    pad_id: int = 0,
    kv_dtype: torch.dtype = torch.float32,
    clock=time.perf_counter,
):
    """Left-padded static batch with a contiguous KV cache. Returns (outputs, first_token_times,
    finish_times, cache_bytes). The batch decodes until every request is done."""
    mc = model.cfg
    B = len(prompts)
    P = max(len(p) for p in prompts)
    max_new = max(p.max_new_tokens for p in params)
    cache = ContiguousKVCache(
        mc.n_layers, B, P + max_new, mc.n_kv_heads or mc.n_heads, mc.d_model // mc.n_heads, dtype=kv_dtype
    )
    valid = torch.zeros(B, P + max_new, dtype=torch.bool)
    ids = torch.full((B, P), pad_id, dtype=torch.long)
    pos = torch.zeros(B, P, dtype=torch.long)
    for b, p in enumerate(prompts):
        pad = P - len(p)
        ids[b, pad:] = torch.tensor(p)
        valid[b, pad:P] = True
        pos[b, pad:] = torch.arange(len(p))
    valid[:, P:] = True
    cache.key_valid = valid
    gens = [torch.Generator().manual_seed(p.seed) if p.seed is not None else None for p in params]
    attn = ContiguousAttention(cache)
    logits = model(ids, pos, attn)[:, -1]
    attn.advance(P)
    outputs: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    first, finish = [None] * B, [None] * B
    for _step in range(max_new):
        now = clock()
        nxt = []
        for b in range(B):
            if done[b]:
                nxt.append(pad_id)
                continue
            tok = sample(logits[b], params[b], gens[b])
            outputs[b].append(tok)
            first[b] = first[b] or now
            if tok in params[b].stop_token_ids or len(outputs[b]) >= params[b].max_new_tokens:
                done[b], finish[b] = True, now
            nxt.append(tok)
        if all(done):
            break
        # Every row advances in lockstep, finished rows included (the cost of static batching).
        step_pos = torch.tensor([[len(prompts[b]) + _step] for b in range(B)])
        logits = model(torch.tensor(nxt)[:, None], step_pos, attn)[:, -1]
        attn.advance(1)
    return outputs, first, finish, cache.bytes
