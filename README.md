# LLM SQL Finetuning

This repository contains code used in the "Deep Learning" course to finetune language models on the LLM-SQL benchmark. It uses Hugging Face Transformers, TRL's SFTTrainer, and a small set of utilities to prepare and run experiments.

## What you'll find

- `src/llm_sql_finetuning`: source code and CLI entrypoint
- `args.yaml`: configuration file with default TrainingArguments and other options
- `data/`: dataset shards and results

## Requirements

- Python 3.11 or later (project requires `>=3.11`)
- A CUDA-capable GPU is recommended for training. If none is available the scripts will run on CPU but training will be very slow.

This project uses `uv` as the build/runtime environment and `pyproject.toml` declares runtime and dev dependencies.

## Quick setup (Windows PowerShell)

1. Clone the repo and enter the directory (you likely already have this):

```powershell
cd C:\path\to\llm-sql-finetuning
```

2. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install `uv` and project dependencies using the build system defined in `pyproject.toml`.

`uv` is a modern Python project runner/build tool used here. Install it with pip:

```powershell
pip install "uv>=0.8.1,<0.9.0"
```

Then install project dependencies (this will read `pyproject.toml`):

```powershell
uv install
```

If you prefer installing with pip directly, you can install the runtime deps from `pyproject.toml` manually:

```powershell
pip install -r <(uv export requirements)
```

Note: On Windows PowerShell the process substitution above may not work; instead run `uv export requirements > requirements.txt` and then `pip install -r requirements.txt`.

## Pre-commit setup

We use `pre-commit` in development. Install and enable it like this:

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This will install hooks defined in `.pre-commit-config.yaml` (if present) and run them on your code.

## Running the project

Basic run (uses the `args.yaml` for configuration):

```powershell
uv run python .\src\llm_sql_finetuning\main.py --config-file args.yaml
```

You can edit `args.yaml` to change dataset paths, model name, or any of the TrainingArguments. The `main.py` constructs a `transformers.TrainingArguments` with these fields set from `args.yaml`. The important keys already present in `args.yaml` are:

- `model_name` (e.g. `google-bert/bert-large-uncased`)
- `dataset_path`
- `output_dir`
- `shots`, `max_seq_len` and `no_eval`
- Training args: `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_train_epochs`, `learning_rate`, `weight_decay`, `warmup_steps`, `logging_dir`, `logging_steps`, `evaluation_strategy`, `save_strategy`, `save_steps`, `eval_steps`, `load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`, `per_device_eval_batch_size`

The script will set `fp16=True` automatically if CUDA is available.

## Example: small local debug run

To run a quick smoke test (no GPU, small dataset):

```powershell
uv run python .\src\llm_sql_finetuning\main.py --config-file args.yaml
```

If you run into CUDA or PyTorch wheel issues, consult the official PyTorch installation instructions and the `tool.uv` index configured in `pyproject.toml`.

## Development tips

- Type checking: `uv run mypy` (project may not include strict type annotations everywhere)
- Linting/formatting: enable `pre-commit` to run formatters (black, isort) if configured

## Notes about `args.yaml`

The file `args.yaml` currently provides default values for every parameter passed into `TrainingArguments` inside `src/llm_sql_finetuning/main.py`. Check or modify the following keys in `args.yaml` if you need to change training behavior:

- `per_device_train_batch_size`
- `per_device_eval_batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs`
- `learning_rate`
- `weight_decay`
- `warmup_steps`
- `logging_dir`
- `logging_steps`
- `evaluation_strategy`
- `save_strategy`
- `save_steps`
- `eval_steps`
- `load_best_model_at_end`
- `metric_for_best_model`
- `greater_is_better`

All these are already present in the default `args.yaml` file shipped with the repo.

## Troubleshooting

- If `uv run python ...` fails, try running the same command directly with the active Python interpreter to see raw errors.
- Ensure your virtualenv is activated and `torch` is installed for the correct CUDA version (if using GPU).

## License and credits

Project created for an academic course. See `pyproject.toml` for author contact.

