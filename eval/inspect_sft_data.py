from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MovieRec LLM SFT JSONL outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sft_movielens_1m"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.data_dir / f"{args.split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= args.limit:
                break
            record = json.loads(line)
            print(json.dumps({k: record[k] for k in ["id", "task", "input", "output"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
