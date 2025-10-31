#!/usr/bin/env python3
import argparse
import sys
import re
from pathlib import Path

HEADER_PATTERN = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE)
FOOTER_PATTERN = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE)


def strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.splitlines()
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if HEADER_PATTERN.search(line):
            start_idx = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        if FOOTER_PATTERN.search(lines[i]):
            end_idx = i
            break

    body = "\n".join(lines[start_idx:end_idx])
    return body


def normalize(text: str) -> str:
    # Ensure LF newlines and trim leading/trailing whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def main():
    p = argparse.ArgumentParser(description="Strip Gutenberg boilerplate and normalize LF/trim.")
    p.add_argument("input", type=Path, help="Path to raw Gutenberg .txt")
    p.add_argument("output", type=Path, help="Path to write cleaned .txt")
    args = p.parse_args()

    raw = args.input.read_text(encoding="utf-8", errors="ignore")
    cleaned = strip_gutenberg_boilerplate(raw)
    cleaned = normalize(cleaned)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(cleaned, encoding="utf-8")
    print(f"Wrote cleaned text: {args.output} (chars={len(cleaned)})")


if __name__ == "__main__":
    sys.exit(main())
