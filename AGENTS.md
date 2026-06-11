# LoRA Matrix Trainer — Agent Instructions

## Project Overview

Python CLI tool for training Anima character LoRA models. Uses kohya-ss training scripts via `accelerate launch`.

## Architecture

- **Entry point**: `scripts/train.py` — unified CLI for single and matrix training modes
- **Training engine**: `sd-scripts/` (kohya-ss/sd-scripts, specifically `anima_train_network.py`)
- **Helpers**: `scripts/command_builder.py`, `scripts/dataset_toml.py` — build training commands and dataset configs
- **Legacy**: `scripts/train_single.py`, `scripts/matrix_trainer.py` — kept for backward compat

## Key Defaults (from proven training runs)

| Parameter | Value |
|-----------|-------|
| network_dim | 20 |
| network_alpha | 1 |
| learning_rate | 0.0002 |
| batch_size | 4 |
| max_steps | 800 |
| optimizer | AdamW8Bit |
| scheduler | cosine |
| mixed_precision | bf16 |
| timestep_sampling | sigmoid |

## Python Environment

- Uses `uv` for dependency management (no pip)
- CUDA 13.0 (cu130) for PyTorch — RTX 50-series GPU
- Run scripts with: `uv run python scripts/train.py ...`

## Dataset Format

Images + matching .txt caption files in a directory:
```
datasets/<name>/img/
  image001.jpg
  image001.txt    ← comma-separated tags
```

Recommended: 12–20 images. `--validate` flag checks this.
