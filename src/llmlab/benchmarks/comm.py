"""All-reduce micro-benchmark (run under torchrun; launched by ``llmlab bench comm``).

Reports algorithm bandwidth ``size / time`` and bus bandwidth ``algbw * 2 (n - 1) / n``
(the NCCL-tests convention for ring all-reduce). On this CPU VM the "interconnect" is
loopback TCP through gloo, so the numbers characterize gloo on one host, not NVLink/IB.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.distributed as dist

SIZES = [2**k for k in range(10, 27, 2)]  # 1 KiB .. 64 MiB


def worker(out_path: str, iters: int = 10) -> None:
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(1)
    rows = []
    for size in SIZES:
        t = torch.ones(size // 4, dtype=torch.float32)
        for _ in range(3):
            dist.all_reduce(t)
        dist.barrier()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            dist.all_reduce(t)
            times.append(time.perf_counter() - t0)
        tt = torch.tensor(sorted(times)[len(times) // 2])
        dist.all_reduce(tt, op=dist.ReduceOp.MAX)  # slowest rank defines completion
        sec = float(tt)
        alg = size / sec
        rows.append(
            {
                "bytes": size,
                "seconds_p50": sec,
                "algbw_GBps": alg / 1e9,
                "busbw_GBps": alg * 2 * (world - 1) / world / 1e9,
            }
        )
    if rank == 0:
        with open(out_path, "w") as fh:
            json.dump({"world_size": world, "backend": "gloo", "rows": rows}, fh)
    dist.destroy_process_group()


if __name__ == "__main__":
    worker(sys.argv[1], int(os.environ.get("COMM_ITERS", "10")))
