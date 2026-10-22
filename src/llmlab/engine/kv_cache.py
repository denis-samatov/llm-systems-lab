"""KV-cache storage: a contiguous per-batch cache and a paged block cache with prefix caching.

Paged layout (as in vLLM's PagedAttention): keys/values of all sequences live in one pool of
fixed-size *blocks*. A sequence owns a *block table* (list of block ids); token at position
``p`` is stored in slot ``block_table[p // block_size] * block_size + p % block_size``.
Memory is allocated per block on demand, so waste is bounded by one partially filled block
per sequence instead of ``max_len - len`` per sequence.

Prefix caching: a *full* block is identified by a chained hash of its tokens and the hash of
the previous block, so equal hashes imply equal token prefixes (up to a 128-bit collision).
Blocks whose reference count drops to zero keep their hash and stay reusable until the
allocator needs them (LRU eviction).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque

import torch


class NoFreeBlocks(RuntimeError):
    pass


def block_hash(parent: bytes | None, tokens: list[int] | tuple[int, ...]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(parent or b"root")
    h.update(torch.tensor(list(tokens), dtype=torch.int64).numpy().tobytes())
    return h.digest()


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = False):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.free: deque[int] = deque(range(num_blocks))
        self.ref = [0] * num_blocks
        self.hash_to_block: dict[bytes, int] = {}
        self.block_to_hash: dict[int, bytes] = {}
        self.evictable: OrderedDict[int, None] = OrderedDict()  # ref==0 cached blocks, LRU order
        self.stats = {"allocated": 0, "evicted": 0, "prefix_hits": 0, "prefix_queries": 0}

    def num_free(self) -> int:
        return len(self.free) + len(self.evictable)

    def num_used(self) -> int:
        return self.num_blocks - self.num_free()

    def allocate(self) -> int:
        if self.free:
            b = self.free.popleft()
        elif self.evictable:
            b, _ = self.evictable.popitem(last=False)
            del self.hash_to_block[self.block_to_hash.pop(b)]
            self.stats["evicted"] += 1
        else:
            raise NoFreeBlocks("KV cache exhausted")
        self.ref[b] = 1
        self.stats["allocated"] += 1
        return b

    def release(self, b: int) -> None:
        if self.ref[b] <= 0:
            raise RuntimeError(f"double free of block {b}")
        self.ref[b] -= 1
        if self.ref[b] == 0:
            if b in self.block_to_hash:
                self.evictable[b] = None
            else:
                self.free.append(b)

    def lookup(self, h: bytes) -> int | None:
        """Return a cached block for hash ``h`` and take a reference on it."""
        self.stats["prefix_queries"] += 1
        b = self.hash_to_block.get(h)
        if b is None:
            return None
        if self.ref[b] == 0:
            self.evictable.pop(b)
        self.ref[b] += 1
        self.stats["prefix_hits"] += 1
        return b

    def register(self, b: int, h: bytes) -> None:
        """Publish a newly filled block under its content hash (first writer wins)."""
        if not self.enable_prefix_caching or b in self.block_to_hash or h in self.hash_to_block:
            return
        self.hash_to_block[h] = b
        self.block_to_hash[b] = h

    def check_invariants(self) -> None:
        free = set(self.free)
        ev = set(self.evictable)
        assert not free & ev, "block both free and evictable"
        for b in range(self.num_blocks):
            if self.ref[b] < 0:
                raise AssertionError(f"negative refcount {b}")
            if self.ref[b] == 0:
                assert b in free or b in ev, f"leaked block {b}"
            else:
                assert b not in free and b not in ev, f"in-use block {b} listed as free"


class PagedKVCache:
    """Physical K/V pool: ``[layers, num_blocks * block_size, kv_heads, head_dim]``."""

    def __init__(
        self,
        n_layers: int,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        shape = (n_layers, num_blocks * block_size, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.block_size = block_size

    @property
    def bytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """K and V for every layer: 2 * L * Hkv * D * bytes."""
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes


class ContiguousKVCache:
    """Per-batch cache ``[layers, B, max_len, kv_heads, head_dim]`` with a shared length.

    Used by the static-batching baseline and by speculative decoding (which needs cheap
    rollback: truncating ``length`` discards rejected draft tokens).
    """

    def __init__(
        self,
        n_layers: int,
        batch: int,
        max_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.k = torch.zeros(n_layers, batch, max_len, n_kv_heads, head_dim, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0
        self.max_len = max_len
        self.key_valid: torch.Tensor | None = None  # [B, max_len] bool, for left padding

    @property
    def bytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()

    def truncate(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError("truncate beyond current length")
        self.length = length
