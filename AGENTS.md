# AGENTS.md

## Overview

This repository is a local working tree for Stanford CS336 Assignment 1: Basics.
The project implements a byte-level BPE tokenizer and a small decoder-only
Transformer language model from scratch, then uses them for the Section 7
experiments on TinyStories and OpenWebText.

The repo is organized around three main workflows:

1. Implement tokenizer / model components in `cs336_basics/`
2. Validate correctness with the official tests in `tests/`
3. Run end-to-end experiments from the repository root

## Environment

- Python: `>=3.11`
- Package manager / command runner: `uv`
- Preferred invocation style: `uv run <script>`

Important:

- Prefer `uv run ...` over plain `python ...`
- `cs336_basics/__init__.py` reads installed package metadata
- Running outside the managed `uv` environment can fail with
  `PackageNotFoundError`

## Repository Layout

- `cs336_basics/`
  Core implementations for BPE, optimizer, Transformer modules, generation,
  and training utilities.
- `tests/`
  Official assignment tests and adapters.
- `data/`
  Raw corpora and generated token binaries such as
  `tinystories_train.bin` / `tinystories_valid.bin`.
- `logs/`
  Experiment records and deliverable templates for Section 7.
- `profile_bpe.py`
  Trains or evaluates a tokenizer on TinyStories / OpenWebText.
- `prepare_token_bin.py`
  Converts a text corpus into a raw 1D token-id `.bin` file using a saved
  tokenizer.
- `run_section7_training.py`
  Runs LM training with Section 7 logging, validation evaluation,
  checkpointing, and run summaries.
- `run_with_json.ps1`
  PowerShell wrapper that expands a JSON config file into normal CLI args
  and then calls `uv run`.
- `prepare_train.json`, `prepare_valid.json`, `section7_train.json`
  Example JSON configs used with `run_with_json.ps1`.
- `实验流程.md`
  Local experiment notes and a command cookbook.

## Core Data Flow

The standard experiment pipeline is:

1. Train a tokenizer with `profile_bpe.py`
2. Save tokenizer files:
   - `*_vocab.json`
   - `*_merges.json`
3. Encode raw text into token binaries with `prepare_token_bin.py`
4. Train the LM with `run_section7_training.py`
5. Inspect outputs in `runs/<experiment_name>_<timestamp>/`

Tokenizer compatibility note:

- `profile_bpe.py` saves tokenizer files using `Tokenizer.save(...)`
- `prepare_token_bin.py` loads them using `Tokenizer.from_files(...)`
- A tokenizer produced by `profile_bpe.py` can therefore be reused directly by
  `prepare_token_bin.py`

## Common Commands

### Sync the environment

```powershell
uv sync --python 3.11
```

### Run tests

```powershell
uv run pytest
```

### Train a TinyStories tokenizer

```powershell
uv run .\profile_bpe.py --mode train --data tinystory
```

### Prepare token binaries directly from CLI

```powershell
uv run .\prepare_token_bin.py --input_txt .\data\TinyStoriesV2-GPT4-train.txt --output_bin .\data\tinystories_train.bin --vocab_json .\tinystories_tokenizer_vocab.json --merges_json .\tinystories_tokenizer_merges.json
```

```powershell
uv run .\prepare_token_bin.py --input_txt .\data\TinyStoriesV2-GPT4-valid.txt --output_bin .\data\tinystories_valid.bin --vocab_json .\tinystories_tokenizer_vocab.json --merges_json .\tinystories_tokenizer_merges.json
```

### Prepare token binaries from JSON configs

```powershell
.\run_with_json.ps1 -Script .\prepare_token_bin.py -Config .\prepare_train.json
```

```powershell
.\run_with_json.ps1 -Script .\prepare_token_bin.py -Config .\prepare_valid.json
```

### Train the Section 7 model from JSON config

```powershell
.\run_with_json.ps1 -Script .\run_section7_training.py -Config .\section7_train.json
```

## Section 7 Notes

- `run_section7_training.py` expects separate train and validation token files
- Output artifacts are written to:
  - `runs/<experiment_name>_<timestamp>/config.json`
  - `runs/<experiment_name>_<timestamp>/metrics.csv`
  - `runs/<experiment_name>_<timestamp>/summary.json`
  - `runs/<experiment_name>_<timestamp>/checkpoints/`
- `metrics.csv` logs:
  - `step`
  - `wallclock_sec`
  - `train_loss`
  - `val_loss`
  - `lr`

Implementation note:

- The script currently sets `d_ff = 4 * d_model`
- This differs from the PDF's TinyStories recommendation of `d_ff = 1344`
- Exact alignment with the handout would require a code change or an extra CLI
  argument

## JSON Wrapper Principle

`run_with_json.ps1` does not change the Python scripts.

It works by:

1. Loading a JSON object with PowerShell
2. Converting each key/value into normal CLI arguments
3. Running `uv run <script> --key value ...`

Example:

```json
{
  "batch_size": 32,
  "device": "cuda:0"
}
```

is expanded into:

```powershell
--batch_size 32 --device cuda:0
```

This means the Python code still uses plain `argparse`; the wrapper only makes
long commands easier to manage and reuse.

## Working Conventions

- Run commands from the repository root
- Keep tokenizer files and `.bin` outputs consistent:
  - same vocab / merges
  - same `special_tokens`
  - matching `dtype`
- When training on `.bin`, the file is treated as a raw 1D token-id stream
- If you switch tokenizer or vocab size, regenerate the `.bin` files

## Windows Notes

- This repository is being used from PowerShell on Windows
- PowerShell line continuation uses the backtick character
- If multi-line commands are inconvenient, prefer single-line commands or the
  JSON-wrapper workflow
