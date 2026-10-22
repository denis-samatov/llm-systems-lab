"""Command line interface.

    llmlab data download                      # fetch + verify RuMedTop3
    llmlab train CONFIG [-o key=value ...]    # single process, or under torchrun for ddp/fsdp2
    llmlab bench {comm,engine,train,quant,spec}
    llmlab report                             # regenerate results/REPORT.md and figures
    llmlab list                               # list runs and benchmark results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llmlab.config import load_config, parse_override


def _cmd_data(args) -> int:
    from llmlab.data import download_rumed

    sums = download_rumed(args.raw_dir)
    print(json.dumps({"verified": sorted(sums)}, indent=2))
    return 0


def _cmd_train(args) -> int:
    from llmlab.train import train

    overrides = dict(parse_override(o) for o in args.override)
    cfg = load_config(args.config, overrides)
    summary = train(cfg, resume=not args.no_resume, max_steps=args.max_steps)
    if summary and summary.get("world_size", 1) >= 1:
        import os

        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(summary, indent=2))
    return 0


def _cmd_bench(args) -> int:
    from llmlab import bench

    return bench.main(args.kind, args.rest)


def _cmd_report(args) -> int:
    from llmlab.report import build_report

    path = build_report(Path(args.results), Path(args.runs))
    print(f"wrote {path}")
    return 0


def _cmd_list(args) -> int:
    runs = Path(args.runs)
    for summary in sorted(runs.glob("*/summary.json")):
        s = json.loads(summary.read_text())
        print(f"run  {s['name']:<28} mode={s['mode']:<6} world={s['world_size']} steps={s['end_step']} "
              f"loss={s['final_loss']:.4f} tok/s={s['tokens_per_s'] or 0:.0f}")
    for res in sorted(Path(args.results).glob("*.json")):
        print(f"bench {res.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llmlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="download and verify the corpus")
    d.add_argument("action", choices=["download"])
    d.add_argument("--raw-dir", default="data/raw")
    d.set_defaults(fn=_cmd_data)

    t = sub.add_parser("train", help="train a model from a YAML config")
    t.add_argument("config")
    t.add_argument("-o", "--override", action="append", default=[], help="dotted key=value override")
    t.add_argument("--no-resume", action="store_true")
    t.add_argument("--max-steps", type=int, default=None, help="stop early (for smoke tests)")
    t.set_defaults(fn=_cmd_train)

    b = sub.add_parser("bench", help="run a benchmark suite")
    b.add_argument("kind", choices=["comm", "engine", "train", "quant", "spec", "profile", "lora"])
    b.add_argument("rest", nargs=argparse.REMAINDER)
    b.set_defaults(fn=_cmd_bench)

    r = sub.add_parser("report", help="regenerate the comparison report")
    r.add_argument("--results", default="results")
    r.add_argument("--runs", default="runs")
    r.set_defaults(fn=_cmd_report)

    ls = sub.add_parser("list", help="list runs and results")
    ls.add_argument("--results", default="results")
    ls.add_argument("--runs", default="runs")
    ls.set_defaults(fn=_cmd_list)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
