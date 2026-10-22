"""Attention contexts plugged into :class:`llmlab.model.GPT` during inference.

* :class:`ContiguousAttention` - one cache row per sequence, shared length (static batch).
* :class:`PagedAttention` - variable-length sequences flattened into one token batch,
  K/V scattered into a block pool via ``slot_mapping`` and gathered back per sequence
  through block tables. This is a readable reference implementation of the PagedAttention
  data path; production kernels fuse the gather into the attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llmlab.engine.kv_cache import ContiguousKVCache, PagedKVCache
from llmlab.model import repeat_kv


class ContiguousAttention:
    def __init__(self, cache: ContiguousKVCache):
        self.cache = cache

    def attend(self, layer: int, q, k, v):
        B, T, H, D = q.shape
        c = self.cache
        start = c.length
        if start + T > c.max_len:
            raise ValueError("contiguous cache overflow")
        c.k[layer, :, start : start + T] = k
        c.v[layer, :, start : start + T] = v
        L = start + T
        keys = repeat_kv(c.k[layer, :, :L], H // k.shape[2]).transpose(1, 2)
        vals = repeat_kv(c.v[layer, :, :L], H // k.shape[2]).transpose(1, 2)
        q_pos = torch.arange(start, L)[:, None]
        k_pos = torch.arange(L)[None, :]
        mask = (k_pos <= q_pos)[None, None]  # [1, 1, T, L]
        if c.key_valid is not None:
            mask = mask & c.key_valid[:, None, None, :L]
        out = F.scaled_dot_product_attention(q.transpose(1, 2), keys, vals, attn_mask=mask)
        return out.transpose(1, 2)

    def advance(self, T: int) -> None:
        self.cache.length += T


@dataclass
class PagedBatch:
    """Metadata for one engine step over S sequences and N = sum(q_lens) new tokens."""

    slot_mapping: torch.Tensor  # [N] physical slot for every new token
    q_lens: list[int]  # new tokens per sequence
    ctx_lens: list[int]  # total tokens per sequence after this step (cached + new)
    key_slots: torch.Tensor  # [S, Lmax] physical slot of each context position (0 for pad)
    key_mask: torch.Tensor  # [S, Qmax, Lmax] causal & valid mask
    q_index: torch.Tensor  # [S, Qmax] index into the flat N tokens (0 for pad)
    q_valid: torch.Tensor  # [S, Qmax]


def build_paged_batch(
    block_tables: list[list[int]], num_computed: list[int], q_lens: list[int], block_size: int
) -> PagedBatch:
    slots = []
    ctx_lens = [c + n for c, n in zip(num_computed, q_lens, strict=True)]
    Lmax, Qmax = max(ctx_lens), max(q_lens)
    key_slots = torch.zeros(len(q_lens), Lmax, dtype=torch.long)
    key_mask = torch.zeros(len(q_lens), Qmax, Lmax, dtype=torch.bool)
    q_index = torch.zeros(len(q_lens), Qmax, dtype=torch.long)
    q_valid = torch.zeros(len(q_lens), Qmax, dtype=torch.bool)
    offset = 0
    for s, (table, computed, n) in enumerate(zip(block_tables, num_computed, q_lens, strict=True)):
        L = computed + n
        pos = torch.arange(L)
        tbl = torch.tensor(table, dtype=torch.long)
        phys = tbl[pos // block_size] * block_size + pos % block_size
        key_slots[s, :L] = phys
        slots.append(phys[computed:L])
        q_pos = torch.arange(computed, L)
        key_mask[s, :n, :L] = pos[None, :] <= q_pos[:, None]
        q_index[s, :n] = torch.arange(offset, offset + n)
        q_valid[s, :n] = True
        offset += n
    return PagedBatch(torch.cat(slots), q_lens, ctx_lens, key_slots, key_mask, q_index, q_valid)


class PagedAttention:
    def __init__(self, cache: PagedKVCache, batch: PagedBatch):
        self.cache = cache
        self.batch = batch

    def attend(self, layer: int, q, k, v):
        # q: [1, N, H, D], k/v: [1, N, Hkv, D] - all new tokens of all sequences, flattened.
        b = self.batch
        _, N, H, D = q.shape
        hkv = k.shape[2]
        kc, vc = self.cache.k[layer], self.cache.v[layer]
        kc[b.slot_mapping] = k[0]
        vc[b.slot_mapping] = v[0]
        keys = repeat_kv(kc[b.key_slots], H // hkv).transpose(1, 2)  # [S, H, Lmax, D]
        vals = repeat_kv(vc[b.key_slots], H // hkv).transpose(1, 2)
        qs = q[0][b.q_index].transpose(1, 2)  # [S, H, Qmax, D]
        mask = b.key_mask[:, None]  # [S, 1, Qmax, Lmax]
        # Padded query rows have an all-False mask; give them one valid key to avoid NaNs,
        # they are dropped below.
        mask = mask | (~b.q_valid)[:, None, :, None] & (torch.arange(mask.shape[-1]) == 0)
        out = F.scaled_dot_product_attention(qs, keys, vals, attn_mask=mask)  # [S, H, Qmax, D]
        out = out.transpose(1, 2)[b.q_valid]  # [N, H, D] in flat order
        return out[None]
