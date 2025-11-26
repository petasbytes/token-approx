#!/usr/bin/env python3
import json
import sys
from pathlib import Path

try:
    import tiktoken  # type: ignore
except Exception as e:
    sys.stderr.write(f"tiktoken not available: {e}\n")
    sys.exit(1)

BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / 'data' / 'processed' / 'datasets' / 'dataset.jsonl'
OUT = BASE / 'data' / 'processed' / 'datasets' / 'preds_tiktoken.jsonl'
METHOD_ID = 'tiktoken_cl100k_base'

ENC = tiktoken.get_encoding('cl100k_base')


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding='utf-8')
    except Exception as e:
        sys.stderr.write(f"failed to read {p}: {e}\n")
        raise


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET, 'r', encoding='utf-8') as f_in, open(OUT, 'w', encoding='utf-8') as f_out:
        for ln in f_in:
            s = ln.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                continue
            source_path = rec.get('source_path')
            if not source_path:
                continue
            abs_path = BASE / source_path
            txt = read_text(abs_path)
            pred_tokens = len(ENC.encode(txt))
            row = {'source_path': source_path,
                   'method_id': METHOD_ID, 'pred_tokens': pred_tokens}
            f_out.write(json.dumps(row, separators=(
                ',', ':'), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
