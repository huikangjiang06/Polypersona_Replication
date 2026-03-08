#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the first N JSON objects from a JSONL file, pretty-formatted."
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        nargs="?",
        default=Path("/home/hj2742/Polypersona_Replication/outputs/experiment_1_synthetic_data/val.json"),
        help="Path to the .jsonl file",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        help="Number of JSON objects to print (default: 10)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for pretty printing (default: 2)",
    )
    args = parser.parse_args()

    if args.num < 1:
        print("Error: --num must be at least 1", file=sys.stderr)
        return 1

    if not args.jsonl_file.is_file():
        print(f"Error: file not found: {args.jsonl_file}", file=sys.stderr)
        return 1

    printed = 0

    try:
        with args.jsonl_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if printed >= args.num:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: skipping invalid JSON on line {line_no}: {e}",
                        file=sys.stderr,
                    )
                    continue

                print(f"--- JSON object {printed + 1} (line {line_no}) ---")
                print(json.dumps(obj, indent=args.indent, ensure_ascii=False))
                print()
                printed += 1

    except OSError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    if printed == 0:
        print("No valid JSON objects found.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())