# token-approx

[![Go Reference](https://pkg.go.dev/badge/github.com/petasbytes/token-approx.svg)](https://pkg.go.dev/github.com/petasbytes/token-approx)
[![Go Report Card](https://goreportcard.com/badge/github.com/petasbytes/token-approx)](https://goreportcard.com/report/github.com/petasbytes/token-approx)
[![License](https://img.shields.io/github/license/petasbytes/token-approx.svg)](LICENSE)

**Evidence-based, lightweight local token approximation for Claude**

`token-approx` helps you train small, calibrated models that locally approximate Claude token counts using basic text features (bytes, runes, words, lines).

> [!NOTE]
> This repo contains the code, data and data-prep tooling behind the [blog post](https://medium.com/@petasbytes/counting-claude-tokens-without-a-tokenizer-e767f2b6e632) **“Counting Claude Tokens Without a Tokenizer”**. You can use it to either:
> - reproduce the experiments from the post, or
> - plug in your own dataset and calibrate a local token approximation model for your own text

___

## Why is this important?

When you’re building agents or tools on top of Claude, **token counts are your core budget and constraint** – they determine:
- how much context you can include
- how often you can call the model
- how much each run costs

For Claude 3+ there’s **no official local tokenizer**, and the usual workarounds aren’t great:
- provider heuristics (e.g. "3.5 characters ≈ 1 token") can be wildly off on real text and across languages
- alternate-provider tokenizers like `tiktoken` systematically under/over-estimate
- calling a remote Token Count API adds network latency, rate limits, extra failure modes, and one more external dependency – *just to count tokens*.

`token-approx` exists to give you a **small, evidence-based, local approximation** instead:
- you calibrate simple linear models on your own text distribution
- you get **fast, pre-send token estimates** that are competitive with, or better than, popular off-the-shelf approximations
- you keep control: everything runs locally, and the whole pipeline (data prep → labeling → modeling) is transparent and reproducible.

This matters most when you care about **tight control over context windows and cost** – e.g. long-running agents, tools that stream lots of user or document text through Claude, or systems where a few percent error in token estimates can mean blowing your budget or silently dropping useful context.

___

## When would you use this?

Use `token-approx` if you:
- **Need local, pre-send token estimates** to manage Claude context windows and budgets
- Use LLMs from a provider (like Anthropic) that **doesn’t ship a modern local tokenizer** but *does* expose token usage counts (via its messages API or a [token-count endpoint](https://platform.claude.com/docs/en/build-with-claude/token-counting))
- Want a small, reproducible pipeline to:
	- build a labeled dataset for your own text distribution
	- fit simple linear models that map local features → token counts
	- easily export them for use in your own application

___

## What’s in this repo?

- **`cmd/token-approx/`** – Go CLI with subcommands `get-data`, `clean`, `split`, `measure`.
- **`data/`** – working directories created and used by the CLI:
  - `data/raw/`, `data/interim/`, `data/processed/samples/`, `data/processed/datasets/`.
- **`notebooks/`** – top-level notebooks + helpers for fitting and comparing models on `data/processed/datasets/dataset.jsonl`.
- **`experiments/`** – pre-prepared datasets and notebooks for the three Gutenberg books used in the [blog post](https://medium.com/@petasbytes/counting-claude-tokens-without-a-tokenizer-e767f2b6e632) (fully wired, no data prep or API calls needed).
- **`scripts/`** – helpers to generate predictions from existing off-the-shelf methods ([Anthropic's suggested heuristic](https://platform.claude.com/docs/en/about-claude/glossary#tokens), `tiktoken`, [Anthropic legacy tokenizer](https://github.com/anthropics/anthropic-tokenizer-typescript)).

___

## How to explore this repo

You have three main options, depending on how much setup you want to do:

1. **Fastest path – inspect pre-prepared experiments**
	- Explore already-run experiments for the example datasets (no additional data prep or API calls required)
		→ see [Experiment & notebooks](#experiments-notebooks-fastest-path)

2. **From-scratch pipeline – reproduce Oliver Twist experiment end-to-end**
	Run the full pipeline on the example Gutenberg dataset:
	 - Prepare data via `token-approx` CLI: `get-data` → `clean` → `split` → `measure` to build `dataset.jsonl`; then use the scripts in `scripts/` to generate baseline predictions for the existing off-the-shelf token approximation methods
	 - Run the experiment using the template notebooks in `notebooks/`
		→ see [From-scratch pipeline](#from-scratch-pipeline-using-example-dataset)

3. **Run on your own data**
   Use the same pipeline structure on your own text:
	 - Prepare data via similar CLI pipeline on your own `.txt` samples: `split` → `measure` to get `dataset.jsonl`; scripts in `scripts/` to generate off-the-shelf method predictions
	 - Run the experiment using the template notebooks in `notebooks/`
		→ see [Use your own data](#use-your-own-data)

___

## Setup

### 1. Clone repo and cd to repo root
```bash
git clone https://github.com/petasbytes/token-approx.git
cd token-approx
```

> [!IMPORTANT]
> - From this point onwards run all commands from repo root to ensure correct data I/O file paths

### 2. Install prerequisites

- Go 1.25+ (required for running the `token-approx` CLI tool)

- Python 3.11+ (required for running notebooks)
- Notebook dependencies and virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Optional: `ANTHROPIC_API_KEY` exported in your environment (for ground-truth labeling using `token-approx measure`)
```bash
export ANTHROPIC_API_KEY=...
```

- Optional: Node.js 20+ & Anthropic legacy tokenizer (for generating token count predictions using the legacy tokenizer):

  If you plan to run the **template notebooks** in `notebooks/`, you’ll need Node + `@anthropic-ai/tokenizer` so `scripts/preds_anth-ts.py` can run:
```bash
# 1) Install Node, e.g. using Homebrew (macOS)
brew install node    # or download from https://nodejs.org

# 2) Install Anthropic legacy tokenizer
npm install @anthropic-ai/tokenizer
```

### 3. Build the CLI

```bash
go build -o token-approx ./cmd/token-approx
./token-approx --help
# If you see module errors: go mod tidy
```

___

## From-scratch pipeline using example dataset

Run the full end-to-end pipeline on a Project Gutenberg text to produce a ready-to-use dataset for the notebooks.

Before proceeding, complete [Setup](#setup) (incl. Node + `@anthropic-ai/tokenizer`).

> [!IMPORTANT]
> Run all commands from repo root

1. Prepare and label the example data (Oliver Twist):
```bash
# 1) Download raw text
./token-approx get-data

# 2) Clean: strip Gutenberg boilerplate, normalize whitespace
./token-approx clean

# 3) Split into roughly equal-sized samples
./token-approx split

# 4) Label samples with ground-truth token counts
export ANTHROPIC_API_KEY=...
./token-approx measure         # calls Anthropic Token Count API (free, rate-limited)
```
	→ Outputs: 
		- raw: `data/raw/oliver-twist_gberg_raw.txt`
		- cleaned: `data/interim/oliver-twist_gberg_clean.txt`
		- samples: `data/processed/samples/oliver-twist_gberg_sample-XXX.txt`
		- labeled dataset: `data/processed/datasets/dataset.jsonl`

2. Generate baseline predictions for existing methods:
```bash
# 1) Generate token predictions using the legacy tokenizer
python scripts/preds_anth-ts.py

# 2) Generate token predictions using 3.5-char-per-token heuristic
python scripts/preds_heuristic.py

# 3) Generate token predictions using tiktoken
python scripts/preds_tiktoken.py
```
	→ Output: three `preds_*.jsonl` files in `data/processed/datasets/`

3. Run the main experiment notebook (`notebooks/01_eda_and_baselines.ipynb`)
```bash
source .venv/bin/activate
jupyter notebook notebooks/01_eda_and_baselines.ipynb
```

This main notebook will:
- load `data/processed/datasets/dataset.jsonl` and the `preds_*.jsonl` files
- compute existing method baselines (heuristic, legacy tokenizer, `tiktoken`, etc.)
- fit and compare single- and multi-feature linear models
- export coefficients to `models/model_coefs.json`

4. Optional: Run the appendix notebook (`notebooks/02_appendix_diagnostics.ipynb`) to explore additional diagnostics and plots.

___

## Use your own data

You can also use `token-approx` as a small **dataset builder + token-labeler** for your own text.

Before proceeding, ensure you have completed the required [setup](#setup) (incl. installation of Node & the legacy Anthropic tokenizer package).

> [!TIP]
> If you're starting with a **single large text file** (i.e. you don't yet have individual text samples) then start at Step 1.
> 
> If you **already have per-sample `.txt` files** then skip to step 2.


1. *(Optional – only if starting with a single large cleaned file)*  
   Place your cleaned text file at `data/interim/<basename>_clean.txt`, then split it into samples:
```bash
./token-approx split
```
	→ Output: multiple `data/processed/samples/<basename>_sample-XXX.txt`

2. Ensure you have one `.txt` file per sample in `data/processed/samples/`

3. Generate labeled dataset and baseline prediction files for the off-the-shelf methods:

  - Compute features and label with token counts:
```bash
export ANTHROPIC_API_KEY=...
./token-approx measure
```
	→ Output: `data/processed/datasets/dataset.jsonl`

  - Generate baseline token-count predictions for the existing off-the-shelf token approximation methods:
```bash
python scripts/preds_anth-ts.py
python scripts/preds_heuristic.py
python scripts/preds_tiktoken.py
```
	→ Output: three `preds_*.jsonl` files in `data/processed/datasets/`

4.  Run the notebooks in `notebooks/`, e.g.:
```bash
source .venv/bin/activate
jupyter notebook notebooks/01_eda_and_baselines.ipynb
```

> [!NOTE]
> **Idempotence for `token-approx measure`**
> - Re-running `measure` will **skip already-measured files** identified by `source_path` in `dataset.jsonl`
> - To re-measure/re-label a single sample, delete its line from the JSONL file and re-run `token-approx measure`

For more details, see [CLI commands](#cli-commands).

___

## Experiments & notebooks (fastest path)

### Pre-prepared experiments (`experiments/`)

> [!INFO]
> The notebooks in `experiments/` can be read/re-run without any additional data prep (as all relevant data has been pre-computed).
> 	→ This means you do **not** need an Anthropic API key, Node, the legacy tokenizer, or to run any CLI commands or prediction scripts.

For each narrative text mentioned in the [blog post](https://medium.com/@petasbytes/counting-claude-tokens-without-a-tokenizer-e767f2b6e632), you’ll find a directory under `experiments/`:
- `experiments/oliver-twist_gberg/`
- `experiments/war-n-peace_gberg/`
- `experiments/les-trois-mousq_gberg/`

Each has ready-to-go:
- Data in `experiments/<book-id>/data/processed/datasets/`
	- `dataset.jsonl`
	- `preds_anth-ts.jsonl`, `preds_heuristic.jsonl`, `preds_tiktoken.jsonl`
- Notebooks (with results): `experiments/<book-id>/notebooks/`
	- `01_eda_and_baselines_<book-id>.ipynb`
	- `02_appendix_diagnostics_<book-id>.ipynb`

To explore a specific book’s experiment:

1. [Setup](#setup) notebook dependencies and virtual environment

2. Open the relevant notebook for that book, e.g.:
```bash
source .venv/bin/activate
jupyter notebook experiments/oliver-twist_gberg/notebooks/01_eda_and_baselines_oliver-twist.ipynb
```

___

## CLI commands

All `token-approx` CLI commands operate relative to the repo root and the `./data` directory.

### `get-data`
- Downloads Oliver Twist from Project Gutenberg
- Writes: `data/raw/oliver-twist_gberg_raw.txt`

### `clean`
- Expects exactly one `data/raw/*_raw.txt`
- Strips Gutenberg boilerplate, normalizes newlines, trims whitespace
- Writes: `data/interim/<book-id>_clean.txt` (e.g. `data/interim/oliver-twist_gberg_clean.txt`)

### `split`
- Expects exactly one `data/interim/*_clean.txt`
- Splits the text into roughly equal-sized (by character/rune) samples
- Writes: `data/processed/samples/<basename>_sample-XXX.txt`

### `measure`
- Requires `ANTHROPIC_API_KEY` in env
- Processes all regular, non-hidden `*.txt` files in `data/processed/samples/` (no recursion)
- For each sample:
	- computes local features: `bytes`, `runes`, `words`, `lines`
	- gets ground-truth `input_tokens` via Anthropic’s Token Count API (free, rate-limited)
- Appends one JSONL object per sample to `data/processed/datasets/dataset.jsonl` with fields:
	- `id`, `model`, `input_tokens`, `features.{bytes,runes,words,lines}`, `source_path`

___

## Troubleshooting

- **Missing API key**
	- `measure` exits early with `missing ANTHROPIC_API_KEY` → set it and re-run
- **No samples found**
	- Ensure `.txt` files exist in `data/processed/samples/` (flat directory, no hidden files, not directories)
- **Multiple inputs for `clean` / `split`**
	- Keep exactly one matching file in `data/raw/*_raw.txt` (for `clean`) or `data/interim/*_clean.txt` (for `split`)
- **Token Count API rate limits**
	- If you hit rate limits, wait and re-run `measure` (see [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/token-counting#pricing-and-rate-limits) for current limits)

___

## Further reading

- **Blog post**
	- [*Counting Claude Tokens Without a Tokenizer*](https://medium.com/@petasbytes/counting-claude-tokens-without-a-tokenizer-e767f2b6e632) – background, experimental design, results tables, and discussion
- **Related project**
	- [`go-agent`](https://github.com/petasbytes/go-agent) – LLM-backed agent in Go that originally motivated this token-approximation project
