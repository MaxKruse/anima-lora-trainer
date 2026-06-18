# LoRA Matrix Trainer — Agent Instructions

## Project Overview

Python CLI tool for training Anima character LoRA models. Runs kohya-ss training **in-process** (imports `AnimaNetworkTrainer` directly, no subprocess or `accelerate launch`).

## Architecture

- **Entry point**: `scripts/train.py` — unified CLI for single and matrix training modes
- **Training engine**: `sd-scripts/anima_train_network.py` (kohya-ss, imported in-process)
- **Core modules**:
  - `scripts/cli_args.py` — argument parser, param parsing (single + matrix ranges)
  - `scripts/constants.py` — default values, model paths, validation limits
  - `scripts/validation.py` — dataset validation (image counts, captions, markers)
  - `scripts/dataset_toml.py` — generates `dataset.toml` configs for kohya-ss
  - `scripts/bucket_rebalance.py` — detects dominant aspect-ratio buckets and redistributes via cropping
  - `scripts/zip_training_data.py` — archives training images into a backup zip
- **Utility scripts**: `infer_batch.py`, `list_loras.py`, `model_verify.py`, `prompt_generator.py`, `tag_extractor.py`, `rename_output_dirs.py`, `setup_env.py`

## Key Defaults (from `scripts/constants.py`)

| Parameter | Value |
|-----------|-------|
| network_dim | 20 |
| network_alpha | 1 |
| learning_rate | 0.0002 |
| epochs | 2 |
| batch_size | 4 |
| max_steps | 800 |
| resolution | 1024 |
| optimizer | AdamW8Bit |
| scheduler | cosine |
| mixed_precision | bf16 |
| timestep_sampling | sigmoid |
| caption_tag_dropout_rate | 0.1 |
| keep_tokens | 1 |

Save interval is `max_steps // epochs` (e.g., 800 steps / 4 epochs = save every 200 steps).

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
