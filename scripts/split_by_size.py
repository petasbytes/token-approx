#!/usr/bin/env python3
import argparse
from pathlib import Path

def chunk_text(text: str, target: int = 32000, max_chars: int = 36000):
    parts = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + target, n)
        end = min(i + max_chars, n)
        # try to end on a paragraph boundary within [target, max]
        k = text.rfind("\n\n", j, end)
        if k == -1 or k <= i:
            k = end if end > i else n
        parts.append(text[i:k].strip())
        i = k
    return parts


def main():
    ap = argparse.ArgumentParser(description="Split a text file into ~size-based parts (runes≈chars), optionally ending on paragraph boundaries")
    ap.add_argument("input", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--base-name", default="gutenberg_clean")
    ap.add_argument("--target", type=int, default=32000)
    ap.add_argument("--max-chars", type=int, default=36000)
    args = ap.parse_args()

    s = args.input.read_text(encoding="utf-8")
    parts = chunk_text(s, args.target, args.max_chars)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, part in enumerate(parts, 1):
        p = args.output_dir / f"{args.base_name}_part_{idx:03d}.txt"
        p.write_text(part, encoding="utf-8")
        print(f"wrote {p} chars={len(part)}")

if __name__ == "__main__":
    main()
