# Experiment: Local Token Approximation for Language Models

A small, reproducible experiment to approximate language model provider tokenizer input tokens (in this case Anthropic's Claude) using simple local text features (bytes, runes, words, lines). Goal: enable rough, on-device token budgeting to improve token-budgeting and agent context window management, while reducing reliance on external token-counting APIs calls and unreliable/imprecise third-party tools/heuristics.

## TL;DR
- Hypothesis: linear relationships between local features and `input_tokens` are strong enough for budgeting.
- Data: 25 samples with features and ground-truth tokens for `claude-3-7-sonnet-latest`.
- Approach A (single feature baseline): `bytes + intercept` on held-out test → MAE ≈ 155 tokens.
- Approach B (linear multi-feature): best subset OLS (`bytes + words`) → MAE ≈ 129 tokens (≈17% better), lower bias.
- Typical test abs error: median ≈ 100, p95 ≈ 250–280 (n=5 test ⇒ indicative only). Add a safety buffer near limits.

## Repo Layout
- `notebooks/01_eda_and_baselines.ipynb` — main narrative, EDA, A vs B selection, exports.
- `notebooks/02_appendix_diagnostics.ipynb` — diagnostics, ablations, learning curves.
- `notebooks/utils.py` — helpers (CV, exporters, plots).
- `output/records.jsonl` — input data (produced upstream).
- `output/model_coefs.json` — exported chosen model (used downstream).
- `output/tables/` and `output/figures/` — artifacts.

## How to Run
1) Python 3.10+
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Ensure data exists at:
```
output/records.jsonl
```
4) Open and run:
- Start with `notebooks/01_eda_and_baselines.ipynb`
- Deep-dive in `notebooks/02_appendix_diagnostics.ipynb`

## What This Enables
- Rough on-device token estimates for budgeting/cost throttling and agent context window management (reducing reliance on external token-counting APIs calls and unreliable/imprecise third-party tools/heuristics).
- Exported coefficients at `output/model_coefs.json` for simple linear prediction in other codebases (e.g., `go-agent`).

## Notes & Scope
- Token relationships are tokenizer- and model-specific; results shown for `claude-3-7-sonnet-latest`.
- Small dataset; use the appendix notebook to see stability checks and where more data would help (e.g. learning curves).
- Favour the simpler model unless larger models deliver clear, robust gains beyond a small margin.
