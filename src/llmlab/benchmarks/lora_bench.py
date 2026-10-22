"""Parameter-efficient fine-tuning on a downstream task: ICD-10 coding of RuMedTop3 complaints.

Compared on the same splits as ``rumed-icd-llm`` (train 4,690 / dev 848 / test 822):

* ``tfidf_lr``    - word 1-2 + char 2-5 n-gram TF-IDF, logistic regression (C=10), fixed.
* ``linear_probe``- frozen pretrained llmlab target model, trainable linear head.
* ``lora_r8``     - PEFT LoRA (r=8, alpha=16) on q/k/v/o projections + head.
* ``full_ft``     - all backbone parameters + head.

The backbone was pretrained only on the *train* complaints (causal LM), so no dev/test text
was seen. Epoch selection uses dev Hit@1; test is evaluated once at the selected epoch.
The point is the engineering trade-off (trainable parameters, optimizer memory, step time,
quality), not state-of-the-art accuracy: the 5.5M-parameter backbone is tiny.
"""

from __future__ import annotations

import copy
import random
import time

import torch
from torch import nn

from llmlab.benchmarks.common import load_model, write_result
from llmlab.data import PAD, load_rumed, load_tokenizer

EPOCHS = 8
MAX_LEN = 128


def _bootstrap_ci(correct: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(correct)
    stats = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def _hits(scores: torch.Tensor, gold: list[int]) -> tuple[list[int], list[int]]:
    top3 = scores.topk(3, dim=-1).indices.tolist()
    h1 = [int(g >= 0 and t[0] == g) for t, g in zip(top3, gold, strict=True)]
    h3 = [int(g >= 0 and g in t) for t, g in zip(top3, gold, strict=True)]
    return h1, h3


def _report(name: str, h1: list[int], h3: list[int], extra: dict) -> dict:
    lo, hi = _bootstrap_ci(h1)
    return {
        "method": name,
        "test_hit1": sum(h1) / len(h1),
        "test_hit1_ci95": [lo, hi],
        "test_hit3": sum(h3) / len(h3),
        **extra,
    }


class Classifier(nn.Module):
    def __init__(self, backbone, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.cfg.d_model, n_classes)

    def forward(self, ids, lengths):
        h = self.backbone.hidden(ids)
        return self.head(h[torch.arange(ids.shape[0]), lengths - 1])


def _encode(tok, texts, pad_id):
    rows = [tok.encode(t).ids[:MAX_LEN] or [pad_id] for t in texts]
    lengths = torch.tensor([len(r) for r in rows])
    ids = torch.full((len(rows), MAX_LEN), pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
    return ids, lengths


@torch.no_grad()
def _scores(model, ids, lengths, bs=128):
    model.eval()
    out = torch.cat([model(ids[i : i + bs], lengths[i : i + bs]) for i in range(0, len(ids), bs)])
    model.train()
    return out


def _optim_bytes(opt) -> int:
    return sum(
        v.numel() * v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)
    )


def _finetune(name, model, data, lr, seed=0):
    (tr_ids, tr_len, tr_y), (dv_ids, dv_len, dv_y), (te_ids, te_len, te_y) = data
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    best, best_state, step_times, log = -1.0, None, [], []
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(tr_ids), generator=g)
        for i in range(0, len(perm), 32):
            idx = perm[i : i + 32]
            t0 = time.perf_counter()
            loss = nn.functional.cross_entropy(model(tr_ids[idx], tr_len[idx]), tr_y[idx])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step_times.append(time.perf_counter() - t0)
        h1, _ = _hits(_scores(model, dv_ids, dv_len), dv_y.tolist())
        dev = sum(h1) / len(h1)
        log.append({"epoch": epoch + 1, "dev_hit1": dev})
        if dev > best:
            best, best_state = dev, copy.deepcopy({k: v for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    h1, h3 = _hits(_scores(model, te_ids, te_len), te_y.tolist())
    steady = step_times[5:]
    return _report(
        name,
        h1,
        h3,
        {
            "dev_hit1_best": best,
            "best_epoch": max(log, key=lambda r: r["dev_hit1"])["epoch"],
            "dev_curve": log,
            "trainable_params": sum(p.numel() for p in params),
            "total_params": sum(p.numel() for p in model.parameters()),
            "optimizer_state_bytes": _optim_bytes(opt),
            "mean_step_s": sum(steady) / len(steady),
            "lr": lr,
        },
    )


def main() -> None:
    from peft import LoraConfig, inject_adapter_in_model
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline

    torch.manual_seed(0)
    train, dev, test = (load_rumed(s) for s in ("train", "dev", "test"))
    codes = sorted({r.code for r in train})
    cid = {c: i for i, c in enumerate(codes)}
    y = {
        n: torch.tensor([cid.get(r.code, -1) for r in recs])
        for n, recs in (("train", train), ("dev", dev), ("test", test))
    }
    rows = []

    # 1) Classical baseline
    t0 = time.perf_counter()
    vec = FeatureUnion(
        [
            ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ]
    )
    clf = make_pipeline(vec, LogisticRegression(C=10, max_iter=2000))
    clf.fit([r.text for r in train], [cid[r.code] for r in train])
    fit_s = time.perf_counter() - t0
    proba = torch.tensor(clf.predict_proba([r.text for r in test]))
    full = torch.full((len(test), len(codes)), -1.0)
    full[:, torch.tensor(clf.classes_)] = proba.float()
    h1, h3 = _hits(full, y["test"].tolist())
    rows.append(_report("tfidf_lr", h1, h3, {"fit_s": fit_s}))

    tok = load_tokenizer("artifacts/tokenizer.json")
    pad = tok.token_to_id(PAD)
    enc = {
        n: _encode(tok, [r.text for r in recs], pad)
        for n, recs in (("train", train), ("dev", dev), ("test", test))
    }
    data = tuple((*enc[n], y[n]) for n in ("train", "dev", "test"))
    # Unknown dev/test codes (-1) can never be predicted; they count as errors.
    rows[0]["unseen_test_codes"] = int((y["test"] < 0).sum())

    base = load_model("runs/target")
    # 2) linear probe
    probe = Classifier(copy.deepcopy(base), len(codes))
    for p in probe.backbone.parameters():
        p.requires_grad_(False)
    rows.append(_finetune("linear_probe", probe, data, lr=3e-3))
    # 3) LoRA
    lora_backbone = inject_adapter_in_model(
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        copy.deepcopy(base),
    )
    rows.append(_finetune("lora_r8", Classifier(lora_backbone, len(codes)), data, lr=2e-3))
    # 4) full fine-tune
    rows.append(_finetune("full_ft", Classifier(copy.deepcopy(base), len(codes)), data, lr=5e-4))
    for r in rows:
        print(f"{r['method']:<13} test Hit@1 {r['test_hit1']:.3f} Hit@3 {r['test_hit3']:.3f}")
    print(f"wrote {write_result('lora', {'epochs': EPOCHS, 'n_classes': len(codes), 'rows': rows})}")


def timing(steps: int = 40, warmup: int = 5) -> None:
    """Step time of probe / LoRA / full fine-tuning in isolation (same batch, same model).

    Kept separate from :func:`main` because step time is sensitive to other load on the host;
    run it on an otherwise idle machine. Quality numbers come from :func:`main`.
    """
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(0)
    base = load_model("runs/target")
    tok = load_tokenizer("artifacts/tokenizer.json")
    texts = [r.text for r in load_rumed("train")][:32]
    ids, lengths = _encode(tok, texts, tok.token_to_id(PAD))
    y = torch.randint(0, 105, (32,))
    variants = {}
    probe = Classifier(copy.deepcopy(base), 105)
    for p in probe.backbone.parameters():
        p.requires_grad_(False)
    variants["linear_probe"] = probe
    variants["lora_r8"] = Classifier(
        inject_adapter_in_model(
            LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            copy.deepcopy(base),
        ),
        105,
    )
    variants["full_ft"] = Classifier(copy.deepcopy(base), 105)
    rows = []
    for name, model in variants.items():
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        times = []
        for i in range(warmup + steps):
            t0 = time.perf_counter()
            nn.functional.cross_entropy(model(ids, lengths), y).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if i >= warmup:
                times.append(time.perf_counter() - t0)
        times.sort()
        rows.append(
            {
                "method": name,
                "median_step_s": times[len(times) // 2],
                "p90_step_s": times[int(0.9 * len(times))],
                "steps": steps,
                "batch": 32,
                "max_len": MAX_LEN,
            }
        )
        print(f"{name:<13} median step {rows[-1]['median_step_s']:.3f} s")
    print(f"wrote {write_result('lora_timing', {'threads': torch.get_num_threads(), 'rows': rows})}")
