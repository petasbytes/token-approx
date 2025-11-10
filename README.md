# token-approx

[![Go Reference](https://pkg.go.dev/badge/github.com/petasbytes/token-approx.svg)](https://pkg.go.dev/github.com/petasbytes/token-approx)
[![Go Report Card](https://goreportcard.com/badge/github.com/petasbytes/token-approx)](https://goreportcard.com/report/github.com/petasbytes/token-approx)
[![License](https://img.shields.io/github/license/petasbytes/token-approx.svg)](LICENSE)

Evidence-based, lightweight token approximation for LLMs

**What**
- `token-approx` CLI tool to prepare a labeled dataset
- Notebooks to fit simple linear models that estimate token counts from local text features (bytes/runes/words/lines)

**Why**
Get reliable, local token estimates so you can:
- Plan context usage before you pay for it
- Make fewer provider API calls
- Reduce budget surprises

**How**
1. Prepare samples:
   - Use the example pipeline: `token-approx get-data` → `clean` → `split`
   - Or drop your own per-sample `.txt` files into `data/processed/samples/`
2. Label ground truth with Anthropic’s Token Count API via `token-approx measure`
3. Run the notebooks to fit baselines, inspect diagnostics, and export coefficients for your application

---

## Quickstart

***Note: run all from repo root for predictable paths (I/O is under `./data/...`)***

**Prereqs:** Go 1.25+, Python 3.13+, `ANTHROPIC_API_KEY` in your env

1) Get the repo and build the CLI
```bash
git clone https://github.com/petasbytes/token-approx.git
cd token-approx

go build -o token-approx ./cmd/token-approx
./token-approx --help
# If you see module errors: go mod tidy
```

2) Run the example pipeline (Oliver Twist)
```bash
./token-approx get-data
./token-approx clean
./token-approx split
export ANTHROPIC_API_KEY=...  # required for measure
./token-approx measure        # calls Anthropic Token Count API (free, rate-limited)
```

3) Install notebook deps and open
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_and_baselines.ipynb
```

---

## Use your own dataset

1. Import your data 1 of 2 ways:
a) Put per-sample `.txt` files in `data/processed/samples/` (no nested folders)
  OR
b) Place one large text file in `data/interim/<basename>_clean.txt`; then run `./token-approx split`
2) Set `ANTHROPIC_API_KEY` (i.e. run `export ANTHROPIC_API_KEY=...`)
3) Run `./token-approx measure` to build `data/processed/datasets/dataset.jsonl`

**Notes re idempotence:**
- Re-running `measure` will skip files already present in `dataset.jsonl` (identified by `source_path`)
- To re-measure a specific file, delete its JSONL entry and rerun `measure`

---

## CLI usage

`get-data`:
- Downloads Oliver Twist (Project Gutenberg)
- Writes: `data/raw/oliver-twist_gberg_raw.txt`

`clean`:
- Operates on exactly one `data/raw/*_raw.txt`
- Strips Gutenberg boilerplate, normalizes newlines, trims whitespace
- Writes `data/interim/<basename>_clean.txt` (e.g. `data/interim/oliver-twist_gberg_clean.txt`)

`split`
- Operates on exactly one `data/interim/*_clean.txt`
- Deterministic chunking: prefers `\n\n` between 32k–36k runes
- Writes `data/processed/samples/<basename>_sample-XXX.txt` (e.g. `data/processed/samples/oliver-twist_gberg_sample-001.txt`, `...sample-002.txt`, etc)

`measure`
- Requires `ANTHROPIC_API_KEY`
- Batches all regular, non-hidden `*.txt` in `data/processed/samples/` (no recursion)
- Computes local features (bytes, runes, words, lines) and calls Anthropic Token Count API (free; rate-limited — see [Anthropic docs](https://docs.claude.com/en/docs/build-with-claude/token-counting))
- Appends one JSON object per line to `data/processed/datasets/dataset.jsonl` with fields: `id, model, input_tokens, features.{bytes, runes, words, lines}, source_path`
- Prints summary showing successful/failed files

Default token counting model: `claude-3-7-sonnet-latest`

---

## Notebooks workflow

- Input: `data/processed/datasets/dataset.jsonl` (produced by `measure`)
- Main Experiment: `notebooks/01_eda_and_baselines.ipynb`
- Appendix: `02_appendix_diagnostics.ipynb`
- Utilities: `notebooks/utils.py` for loading/validation, OLS/Ridge/ElasticNet baselines, diagnostics, learning curves, and exporting coefficients
- Outputs: coefficients → `models/model_coefs.json` (created in `notebooks/01_eda_and_baselines.ipynb`)

---

## Troubleshooting

- Missing API key: `measure` exits early with `missing ANTHROPIC_API_KEY`
- No samples: ensure `.txt` files exist in `data/processed/samples/` (regular, non-hidden; not directories)
- Multiple inputs for clean/split: keep exactly one matching file in the expected directory
- Rate limits: if Token Count API rate limits are hit, wait and retry (see [Anthropic docs](https://docs.claude.com/en/docs/build-with-claude/token-counting))
- Network hiccups (`get-data`/`measure`): re-run; transient failures can occur

---

## FAQ

- Can I skip `get-data`? Yes. Provide your own samples in `data/processed/samples/` and run `measure`
- What model is used for counting? `claude-3-7-sonnet-latest` by default (changeable in code)
- Can I re-measure a file? Delete its line in `data/processed/datasets/dataset.jsonl` and rerun `measure`
- Is the Token Count API free? Yes; it’s free but rate-limited (see [Anthropic docs](https://docs.claude.com/en/docs/build-with-claude/token-counting))
