#!/usr/bin/env python3
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / 'data' / 'processed' / 'datasets' / 'dataset.jsonl'
OUT = BASE / 'data' / 'processed' / 'datasets' / 'preds_heuristic.jsonl'
METHOD_ID = 'heuristic_3.5-runes-per-token'


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
            features = rec.get('features') or {}
            runes = features.get('runes')
            source_path = rec.get('source_path')
            if source_path is None or runes is None:
                continue
            pred = float(runes) / 3.5
            row = {'source_path': source_path,
                   'method_id': METHOD_ID, 'pred_tokens': pred}
            f_out.write(json.dumps(row, separators=(
                ',', ':'), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
