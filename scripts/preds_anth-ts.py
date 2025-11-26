#!/usr/bin/env python3
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / 'data' / 'processed' / 'datasets' / 'dataset.jsonl'
OUT = BASE / 'data' / 'processed' / 'datasets' / 'preds_anth-ts.jsonl'
JS = BASE / 'scripts' / 'anth-ts_count.js'
METHOD_ID = 'anth-ts_default'


def main() -> None:
    node = shutil.which('node')
    if node is None:
        sys.stderr.write('node not found in PATH\n')
        sys.exit(1)
    if not JS.exists():
        sys.stderr.write(f'JS helper missing: {JS}\n')
        sys.exit(1)

    # Collect source paths (use relative paths end-to-end)
    rel_paths = []
    with open(DATASET, 'r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                continue
            sp = rec.get('source_path')
            if not sp:
                continue
            rel_paths.append(sp)

    if not rel_paths:
        sys.stderr.write('no source_path entries found in dataset\n')
        sys.exit(1)

    # Run Node once with all relative paths; ensure CWD is project root
    proc = subprocess.run([node, str(JS), *rel_paths],
                          capture_output=True, text=True, cwd=str(BASE))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)

    # Parse stdout lines into a mapping by relative path
    results = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        sp = obj.get('source_path')
        if not sp:
            continue
        n = obj.get('pred_tokens')
        if isinstance(n, (int, float)):
            results[sp] = int(n)

    # Write outputs in dataset order (relative paths only)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f_out:
        for rel in rel_paths:
            n = results.get(rel)
            if n is not None:
                row = {'source_path': rel,
                       'method_id': METHOD_ID, 'pred_tokens': int(n)}
                f_out.write(json.dumps(row, separators=(
                    ',', ':'), ensure_ascii=False) + '\n')

    # Fail if some paths had no prediction
    missing = [rel for rel in rel_paths if rel not in results]
    if missing:
        sys.stderr.write(
            f"error: {len(missing)} paths had no anth-ts prediction; "
            f"examples: {missing[:5]}\n"
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
