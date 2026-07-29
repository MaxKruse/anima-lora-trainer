# LoRA Matrix Trainer

Train Anima character LoRA models via a lightweight CLI wrapper around [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — single runs or matrix hyperparameter sweeps.

## Prerequisites

- **Python 3.10+** with [uv](https://docs.astral.sh/uv/)
- **NVIDIA GPU** (RTX 30-series or newer, CUDA-enabled)
- **Base model files** (see [Model Setup](#model-setup) below)

## Installation

```bash
# Clone with --recursive to include the sd-scripts submodule
git clone --recursive https://github.com/MaxKruse/lora-matrix-trainer.git
cd lora-matrix-trainer

# If you already cloned without --recursive, initialize the submodule:
git submodule update --init --recursive
```

### Environment Setup

`uv` manages the virtual environment automatically. Run:

```bash
# Detect GPU and generate pyproject.toml with correct CUDA index (cu128 or cu130)
uv run python scripts/setup_env.py

# Install dependencies into .venv (creates it if missing)
uv sync
```

On Windows, activate the venv manually if needed:
```bash
.venv\Scripts\activate
```

On Linux/macOS:
```bash
source .venv/bin/activate
```

Alternatively, prefix commands with `uv run` to use the venv without activating:
```bash
uv run python scripts/train.py --validate --dataset datasets/my-char/img
```

### CLI Commands

After `uv sync`, two commands are registered and available from the venv:

| Command | Description |
|---------|-------------|
| `lora-train` | Train LoRAs (single or matrix mode) |
| `lora-evaluate` | Run inference on trained LoRAs for comparison |

Use them with `uv run` (no activation needed):
```bash
uv run lora-train --validate --dataset datasets/my-char/img
uv run lora-train --dataset datasets/my-char/img --name MyChar
uv run lora-evaluate --dataset datasets/my-char/img --output datasets/my-char/out
```

Or activate the venv first and call them directly:
```bash
# Windows
.venv\Scripts\activate
lora-train --validate --dataset datasets/my-char/img

# Linux/macOS
source .venv/bin/activate
lora-train --validate --dataset datasets/my-char/img
```

### Model Setup

Download these base models into the `models/` directory:

| Model | Directory | File | Source |
|-------|-----------|------|--------|
| Diffusion | `models/diffusion_model/` | `anima-base-v1.0.safetensors` | [Anima release](https://huggingface.co/SkT/Anima) |
| Text Encoder | `models/text_encoder/` | `qwen_3_06b_base.safetensors` | [Qwen3 0.6B Base](https://huggingface.co/Qwen/Qwen3-0.6B) |
| VAE | `models/vae/` | `qwen_image_vae.safetensors` | Included with Anima release |

The `sd-scripts` submodule (kohya-ss training engine) is included automatically via `--recursive` clone.

## Quick Start

```bash
# 1. Validate your dataset (mandatory)
uv run python scripts/train.py --validate --dataset datasets/froot/img

# 2. Train a single LoRA (uses proven defaults)
uv run python scripts/train.py --mode single --dataset datasets/froot/img --name Froot-Anima

# 3. Matrix sweep (all permutations)
uv run python scripts/train.py --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,20,32 --alpha 1,20
```

**Validation is mandatory.** Training will refuse to start until you've run `--validate` on the dataset.

## Dataset Requirements

Place images (`.jpg`, `.png`, etc.) and matching `.txt` caption files in a directory:

```
datasets/froot/img/
  image001.jpg
  image001.txt    ← comma-separated tags, trigger word last
  image002.jpg
  image002.txt
  ...
```

### Multiple Outfits / Variations

Subdirectories are scanned automatically. Each folder with images becomes a separate training subset:

```
datasets/character/img/
  base/            ← base character images
  outfit1/         ← outfit variation 1
  outfit2/         ← outfit variation 2
```

Each subdirectory gets its own `[[datasets.subsets]]` entry in the generated `dataset.toml`.

**Recommended:** 12–20 total images (across all folders) spanning different poses, outfits, and aspect ratios.

### Validate Before Training (Mandatory)

```bash
uv run python scripts/train.py --validate --dataset datasets/froot/img
```

Checks image counts, caption coverage, and per-folder limits. Warnings are non-blocking — only hard errors (missing directory, no images, no captions) prevent training.

## Training Modes

### Single Run

Train one LoRA with fixed parameters. All flags default to values tuned for character training on Anima.

```bash
uv run python scripts/train.py --dataset datasets/froot/img --name Froot-Anima

# custom params
uv run python scripts/train.py --dataset datasets/froot/img --name Froot-Anima --max-steps 600 --lr 0.0002 --bs 4 --network-dim 8 --alpha 1

# disable bucket rebalance
uv run python scripts/train.py --dataset datasets/froot/img --name Froot-Anima --no-rebalance-buckets
```

### Matrix Run

Specify comma-separated values for any parameter — all permutations are trained sequentially.

```bash
uv run python scripts/train.py --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,20,32 --alpha 1,16,20 --lr 0.0001,0.0002
```

This generates 3 × 3 × 2 = **18 training runs**.

**Auto-resume**: Interrupted matrix runs automatically resume on next invocation — completed permutations are skipped, incomplete runs pick up from the latest accelerator state.

**Cancel** a running job:
```bash
# Single mode: touch the cancel file
echo > jobs/<lora-name>.cancel

# Matrix mode: cancel a specific permutation
echo > jobs/matrix-<idx>.cancel
```

## CLI Reference

### Required

| Flag | Description |
|------|-------------|
| `--dataset`, `-d` | Path to training images directory |

### Mode

| Flag | Description |
|------|-------------|
| `--mode`, `-m` | `single` or `matrix` [default: `single`] |
| `--validate` | Validate dataset and exit |

### Output

| Flag | Description |
|------|-------------|
| `--name`, `-n` | LoRA output name (default: dataset folder name) |
| `--output`, `-o` | Output directory base |

### Training Parameters

All accept comma-separated values in matrix mode.

| Flag | Default | Description |
|------|---------|-------------|
| `--network-dim` | 8 | LoRA rank (dimension) |
| `--alpha`, `-a` | 1 | LoRA alpha (effective scale = alpha/dim) |
| `--learning-rate`, `--lr` | 0.0002 | Learning rate |
| `--batch-size`, `--bs` | 4 | Training batch size |
| `--max-steps`, `--ss` | auto | Max training steps (auto from batch_size: bs4=600, bs3=800, bs2=1000, bs1=1600) |
| `--optimizer` | AdamW8Bit | Optimizer type |
| `--scheduler`, `-s` | cosine | LR scheduler |
| `--resolution` | 1024 | Image resolution (768–1024) |
| `--repeats`, `-r` | auto | Override num_repeats (auto from image count) |
| `--mixed-precision` | bf16 | fp16 / bf16 / no |
| `--timestep-sampling` | sigmoid | sigma / uniform / sigmoid / shift / flux_shift |
| `--caption-dropout` | 0.1 | Caption tag dropout rate (0.0–1.0) |
| `--keep-tokens` | 1 | Keep first N tokens from caption shuffle |
| `--no-gradient-checkpointing` | off | Disable gradient checkpointing |
| `--no-cache-latents` | off | Disable latent caching |
| `--cache-text-encoder` | off | Enable text encoder caching |
| `--rebalance-buckets` | on | Detect dominant bucket skew and add random-crop augmented samples |
| `--no-rebalance-buckets` | off | Disable bucket rebalancing |
| `--bucket-dominance-threshold` | 0.20 | Dominant bucket share that triggers rebalance |
| `--bucket-rebalance-max-aug` | 64 | Max augmented crops for rebalance |
| `--bucket-rebalance-seed` | 42 | RNG seed for rebalance crop generation |

### Matrix

| Flag | Description |
|------|-------------|
| `--resume` | Explicit resume flag (auto-resume is default — completed perms are always skipped) |

## Default Parameters (Character Training)

These defaults are tuned for single-character LoRA training on Anima:

| Parameter | Value | Why |
|-----------|-------|-----|
| `network_dim` | 8 | Compact rank - enough for character identity without overfitting |
| `network_alpha` | 1 | Effective scale 0.125 — lets you prompt detail in/out |
| `learning_rate` | 0.0002 | Aggressive but stable with cosine decay |
| `batch_size` | 4 | Balanced gradient signal vs VRAM (~17GB), sweet spot for default params |
| `max_steps` | 600 | Auto-scaled from batch_size (see below) |
| `optimizer` | AdamW8Bit | Memory-efficient, no convergence sacrifice |
| `scheduler` | cosine | Smooth decay from peak LR to near-zero |
| `mixed_precision` | bf16 | Required for Anima/Qwen3 |
| `timestep_sampling` | sigmoid | Focuses on mid-to-high noise levels |
| `caption_tag_dropout` | 0.1 | 10% random tag drops → better generalization |
| `keep_tokens` | 1 | Preserves trigger word from caption shuffle |
| `rebalance_buckets` | on | Reduces bucket skew via random-crop augmentation |

### Auto max_steps by Batch Size

The max_steps value auto-adjusts based on batch_size so the total "theoretical" training stays stable across configurations. Scaling is slightly less than linear:

| Batch Size | Auto max_steps |
|------------|----------------|
| 4 (default) | 600 |
| 3 | 800 |
| 2 | 1000 |
| 1 | 1600 |

Override with `--max-steps` to set a fixed value.

### Repeats Calculation

The `num_repeats` value is auto-calculated so that `(num_images × repeats) / batch_size` falls in the 10-15 range (target ~12 steps per epoch). This ensures no single epoch dominates the training statistically - variety is king. Override with `--repeats`.

### Style Training Tips

For style LoRAs (not character-specific), adjust these parameters:

- **Lower learning rate**: Use ~75% of the default (`--lr 0.00015`)
- **More image variety**: 20-40+ diverse images covering the style broadly
- **Higher network_dim**: Consider 16-32 for complex styles
- The auto max_steps and repeats logic still applies based on your batch_size

### Output Overwrite Protection

If the output model file (`{name}.safetensors`) already exists at the target path, training will abort to prevent accidental overwrites. Remove the existing file or use `--name` / `--output` to choose a different path.

## Output Structure

Single run output:
```
datasets/froot/out/
  .work/
    dataset.toml              ← generated dataset config
    job_manifest.json         ← progress/status
    Froot-Anima-step00000300.safetensors   ← 50% checkpoint
    Froot-Anima-step00000450.safetensors   ← 75% checkpoint
    Froot-Anima.safetensors   ← final checkpoint
    Froot-Anima-state/        ← accelerator state (for resume)
    training-data.zip         ← backup of training data
  Froot-Anima.safetensors     ← copied final model
  Froot-Anima_training_chart.png  ← loss + LR training chart
  training_config.json        ← reproducibility artifact (params + exact CLI command)
```

Matrix runs create subdirectories per permutation and a top-level `manifest.json`:
```
datasets/froot/out/
  manifest.json               ← overall matrix manifest
  training_config.json        ← master matrix config (CLI command + param ranges)
  lr-0.0001xbs-2/
    training_config.json      ← per-permutation config
    dataset.toml
    job_manifest.json
    ...
  lr-0.0001xbs-4/
    training_config.json
    ...
  Froot-Anima-lr-0.0001-bs-2.safetensors  ← copied final models
  Froot-Anima-lr-0.0001-bs-4.safetensors
  ...
```

Each `training_config.json` contains the exact CLI command (`cli_command` field), all effective parameters, and (for matrix) the full param ranges. Copy-paste the CLI command to reproduce the exact run.

Each permutation directory also contains a `*_training_chart.png` with loss and LR curves.

## Project Structure

```
scripts/
  train.py                  ← CLI entry point + training runner
  cli_args.py               ← argument parsing and param conversion
  constants.py              ← defaults, model paths, validation limits
  validation.py             ← dataset validation and marker management
  bucket_rebalance.py       ← bucket skew detection and crop augmentation
  dataset_toml.py           ← kohya-ss dataset config generation
  zip_training_data.py      ← training data zip backups
  infer_batch.py            ← batch inference on existing LoRAs
  prompt_generator.py       ← generate test prompts from tags
  tag_extractor.py          ← extract tags from .txt captions
  model_verify.py           ← verify/download model files
  setup_env.py              ← environment setup helper
  list_loras.py             ← print LoRA tags for prompt use
  legacy/                   ← deprecated migration scripts
sd-scripts/                 ← kohya-ss/sd-scripts (training engine)
models/                     ← downloaded base models
datasets/                   ← training datasets
tests/                      ← test suite
```

## Implementation Notes

- Training runs in-process by importing `anima_train_network.py` from `sd-scripts`
- A full kohya argument namespace is built from upstream defaults, then wrapper values override
- `--validate` writes a `.validation.json` marker — training checks for this before starting
- `--rebalance-buckets` (on by default) detects when one aspect-ratio bucket dominates and redistributes via random-crop augmentation into adjacent buckets
- Checkpoints saved at 50% and 75% of training (final model always saved)
- Auto max_steps: batch_size=4→600, bs=3→800, bs=2→1000, bs=1→1600 (slightly less than linear scaling)
- Auto repeats: `(num_images × repeats) / batch_size` targets ~12 steps per epoch (10-15 range)
- Auto-resume: incomplete runs are detected via `job_manifest.json` status and accelerator state dirs
- Progress is tracked via `job_manifest.json` (single mode) or `manifest.json` (matrix mode)
- `training_config.json` artifact written after successful training for reproducibility
- Output overwrite protection: aborts if `{name}.safetensors` already exists at the output path
- Per-step metrics (loss + LR) captured during training and rendered as an ASCII chart in the terminal after completion
- A PNG training chart (`{name}_training_chart.png`) is saved in the output folder alongside the model

## License

This project is licensed under the [MIT License](LICENSE).

The `sd-scripts/` submodule (kohya-ss/sd-scripts) is licensed under the [Apache License 2.0](sd-scripts/LICENSE.md). It is included as a git submodule and must be initialized with `git submodule update --init --recursive`.
