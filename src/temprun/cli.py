"""Tiny CLI dispatcher for the most common commands. Lets `temprun-train` be
pip-installed as a console_script."""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="temprun", description="CLI helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="Print version")

    p_train = sub.add_parser("train", help="Train from a YAML config")
    p_train.add_argument("--config", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained checkpoint")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--base-model", default=None)
    p_eval.add_argument("--eval-jsonl", required=True)
    p_eval.add_argument("--mode", choices=["logits", "generate"], default="logits")
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.add_argument("--max-length", type=int, default=2048)
    p_eval.add_argument("--out", default=None)

    p_inf = sub.add_parser("infer", help="Run inference on a test dir")
    p_inf.add_argument("--checkpoint", required=True)
    p_inf.add_argument("--base-model", default=None)
    p_inf.add_argument("--test-dir", required=True)
    p_inf.add_argument("--out", required=True)
    p_inf.add_argument("--mode", choices=["logits", "generate"], default="logits")
    p_inf.add_argument("--batch-size", type=int, default=16)
    p_inf.add_argument("--max-length", type=int, default=2048)

    args = parser.parse_args()

    if args.cmd == "version":
        from temprun import __version__
        print(__version__)
        return

    if args.cmd == "train":
        from temprun.train import train
        from temprun.utils import load_config
        cfg = load_config(args.config)
        train(cfg)
        return

    if args.cmd == "evaluate":
        from temprun.utils import load_config
        sys.path.insert(0, "scripts")
        import evaluate as ev
        ev.main(args)
        return

    if args.cmd == "infer":
        from temprun.utils import load_config
        sys.path.insert(0, "scripts")
        import infer as inf
        inf.main(args)
        return


if __name__ == "__main__":
    main()
