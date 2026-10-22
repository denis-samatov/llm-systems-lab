"""Typed, validated experiment configuration.

Every experiment is a YAML file parsed into :class:`ExperimentConfig`. Unknown keys are
rejected so that a typo cannot silently fall back to a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Strict):
    vocab_size: int = Field(4096, ge=64)
    d_model: int = Field(256, ge=16)
    n_layers: int = Field(6, ge=1)
    n_heads: int = Field(8, ge=1)
    n_kv_heads: int | None = Field(None, ge=1, description="None = multi-head attention; < n_heads = GQA")
    ffn_mult: float = Field(8 / 3, gt=0)
    max_seq_len: int = Field(256, ge=8)
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    @model_validator(mode="after")
    def _check_heads(self) -> ModelConfig:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        kv = self.n_kv_heads or self.n_heads
        if self.n_heads % kv:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dim must be even for RoPE")
        return self


class DataConfig(_Strict):
    corpus: Literal["rumed"] = "rumed"
    raw_dir: str = "data/raw"
    tokenizer_path: str = "artifacts/tokenizer.json"
    seq_len: int = Field(128, ge=8)


class OptimConfig(_Strict):
    lr: float = Field(3e-3, gt=0)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)
    warmup_steps: int = Field(20, ge=0)
    weight_decay: float = Field(0.1, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float | None = 1.0


class ParallelConfig(_Strict):
    mode: Literal["single", "ddp", "fsdp2"] = "single"
    # FSDP2: reshard parameters after forward (ZeRO-3 like). False keeps them gathered until
    # backward finishes (ZeRO-2 like): less communication, more memory.
    reshard_after_forward: bool = True


class TrainConfig(_Strict):
    steps: int = Field(200, ge=1)
    micro_batch_size: int = Field(16, ge=1)
    grad_accum: int = Field(1, ge=1)
    precision: Literal["fp32", "bf16"] = "fp32"
    seed: int = 1234
    eval_every: int = Field(50, ge=1)
    eval_batches: int = Field(8, ge=1)
    checkpoint_every: int = Field(50, ge=1)
    keep_checkpoints: int = Field(2, ge=1)
    log_every: int = Field(10, ge=1)
    threads_per_rank: int | None = Field(None, ge=1)


class ExperimentConfig(_Strict):
    name: str
    output_dir: str = "runs"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    parallel: ParallelConfig = ParallelConfig()
    train: TrainConfig = TrainConfig()
    mlflow: bool = False

    @model_validator(mode="after")
    def _check_lengths(self) -> ExperimentConfig:
        if self.data.seq_len > self.model.max_seq_len:
            raise ValueError("data.seq_len exceeds model.max_seq_len")
        return self

    def global_batch_size(self, world_size: int) -> int:
        return self.train.micro_batch_size * self.train.grad_accum * world_size

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def load_config(path: str | Path, overrides: dict | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return ExperimentConfig.model_validate(raw)


def parse_override(text: str) -> tuple[str, object]:
    """Parse ``a.b=value`` with YAML typing for the value (``3``, ``true``, ``fsdp2``)."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)
