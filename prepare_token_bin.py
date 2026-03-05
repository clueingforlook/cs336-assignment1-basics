#!/usr/bin/env python3
"""Stream text -> token IDs (.bin) for LM training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cs336_basics.bpe import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a text file into token IDs (.bin).")
    parser.add_argument("--input_txt", type=str, required=True)
    parser.add_argument("--output_bin", type=str, required=True)
    parser.add_argument("--vocab_json", type=str, required=True)
    parser.add_argument("--merges_json", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "int32", "int64"])
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Pass no values to disable, e.g. --special_tokens",
    )
    parser.add_argument("--log_every_lines", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_every_lines <= 0:
        raise ValueError("log_every_lines must be positive.")

    dtype_map = {
        "uint16": np.uint16,
        "int32": np.int32,
        "int64": np.int64,
    }
    np_dtype = dtype_map[args.dtype]

    special_tokens = args.special_tokens
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_json,
        merges_filepath=args.merges_json,
        special_tokens=special_tokens,
    )

    input_path = Path(args.input_txt)
    output_path = Path(args.output_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_lines = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as in_f, output_path.open("wb") as out_f:
        for line in in_f:
            ids = tokenizer.encode(line)
            if ids:
                np.asarray(ids, dtype=np_dtype).tofile(out_f)
                total_tokens += len(ids)
            total_lines += 1
            if total_lines % args.log_every_lines == 0:
                print(f"encoded lines={total_lines} tokens={total_tokens}")

    meta = {
        "input_txt": str(input_path),
        "output_bin": str(output_path),
        "dtype": args.dtype,
        "total_lines": total_lines,
        "total_tokens": total_tokens,
        "special_tokens": special_tokens,
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Finished. output={output_path} total_tokens={total_tokens}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
