# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Personal ML learning repository following a 6-week roadmap: PyTorch training loops → Transformer/GPT from scratch → LoRA fine-tuning → DPO alignment + vLLM serving → RAG with reranking/evaluation → ReAct agents. See `PLAN.md` for the full roadmap (written in Vietnamese).

## Environment

Notebooks use the `llm_env` kernel (Python 3.11). Activate it before running:

```bash
conda activate llm_env   # or: source llm_env/bin/activate
jupyter lab
```

Run a single notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute LLM/instruction_tuning.ipynb
```

## Repository Structure

```
LLM/          # Week 3+ labs: instruction tuning, fine-tuning, serving
  data/       # Training datasets (Alpaca-style JSON)
NLP/          # Week 1–2 labs: PyTorch training loop, nanoGPT
PLAN.md       # 6-week learning roadmap with per-week lab specs and minimum targets
```

New labs should follow the roadmap's week structure — create `LLM/<lab>.ipynb` or `NLP/<lab>.ipynb` as appropriate.

## Lab Conventions

- Each notebook should be self-contained: install deps in the first cell, load data, train, and evaluate.
- Datasets live in `<module>/data/`; use relative paths from the notebook file.
- The instruction tuning dataset format (`LLM/data/instruction_tuning.json`) is Alpaca-style: `{"instruction": ..., "input": ..., "output": ...}` records.
- Minimum target for each lab is defined in `PLAN.md` — meet that before expanding scope.
