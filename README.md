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
uv run lora-train --mode single --dataset datasets/my-char/img --name MyChar
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

Train one LoRA with fixed parameters. All flags default to values from proven training runs.

```bash
uv run python scripts/train.py --mode single --dataset datasets/froot/img --name Froot-Anima

# custom params
uv run python scripts/train.py --mode single --dataset datasets/froot/img --name Froot-Anima --max-steps 500 --lr 0.0002 --bs 4 --network-dim 20 --alpha 1

# bucket skew rebalance (random-crop augmentation)
uv run python scripts/train.py --mode single --dataset datasets/froot/img --name Froot-Anima --rebalance-buckets
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
| `--network-dim` | 20 | LoRA rank (dimension) |
| `--alpha`, `-a` | 1 | LoRA alpha (effective scale = alpha/dim) |
| `--learning-rate`, `--lr` | 0.0002 | Learning rate |
| `--batch-size`, `--bs` | 4 | Training batch size |
| `--max-steps`, `--ss` | 800 | Max training steps |
| `--epochs` | 2 | Number of epochs (checkpoint interval = max_steps / epochs) |
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
| `--rebalance-buckets` | off | Detect dominant bucket skew and add random-crop augmented samples |
| `--bucket-dominance-threshold` | 0.20 | Dominant bucket share that triggers rebalance |
| `--bucket-rebalance-max-aug` | 64 | Max augmented crops for rebalance |
| `--bucket-rebalance-seed` | 42 | RNG seed for rebalance crop generation |

### Matrix

| Flag | Description |
|------|-------------|
| `--resume` | Explicit resume flag (auto-resume is default — completed perms are always skipped) |

## Default Parameters (Proven Recipe)

These defaults come from successful training runs on real character datasets:

| Parameter | Value | Why |
|-----------|-------|-----|
| `network_dim` | 20 | Sweet spot for character detail (16–32 range) |
| `network_alpha` | 1 | Effective scale 0.05 — lets you prompt detail in/out |
| `learning_rate` | 0.0002 | Aggressive but stable with cosine decay |
| `epochs` | 2 | Balanced coverage without overfitting |
| `batch_size` | 4 | Balanced gradient signal vs VRAM (~17GB) |
| `max_steps` | 800 | Enough for convergence, checkpoints every 400 steps (800/2) |
| `optimizer` | AdamW8Bit | Memory-efficient, no convergence sacrifice |
| `scheduler` | cosine | Smooth decay from peak LR to near-zero |
| `mixed_precision` | bf16 | Required for Anima/Qwen3 |
| `timestep_sampling` | sigmoid | Focuses on mid-to-high noise levels |
| `caption_tag_dropout` | 0.1 | 10% random tag drops → better generalization |
| `keep_tokens` | 1 | Preserves trigger word from caption shuffle |

## Output Structure

Single run output:
```
datasets/froot/out/
  .work/
    dataset.toml              ← generated dataset config
    job_manifest.json         ← progress/status
    Froot-Anima-step00000400.safetensors
    Froot-Anima-step00000800.safetensors
    Froot-Anima.safetensors   ← final checkpoint
    Froot-Anima-state/        ← accelerator state (for resume)
    training-data.zip         ← backup of training data
  Froot-Anima.safetensors     ← copied final model
```

Matrix runs create subdirectories per permutation and a top-level `manifest.json`:
```
datasets/froot/out/
  manifest.json               ← overall matrix manifest
  lr-0.0001xsteps-800/        ← permutation working dir
  lr-0.0001xsteps-1200/
  lr-0.0002xsteps-800/
  lr-0.0002xsteps-1200/
  Froot-Anima-lr-0.0001-steps-800.safetensors  ← copied final models
  Froot-Anima-lr-0.0001-steps-1200.safetensors
  ...
```

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
- `--rebalance-buckets` detects when one aspect-ratio bucket dominates and redistributes via random-crop augmentation into adjacent buckets
- Checkpoint interval is `max_steps // epochs` (e.g., 800 steps / 2 epochs = save every 400 steps)
- Auto-resume: incomplete runs are detected via `job_manifest.json` status and accelerator state dirs
- Progress is tracked via `job_manifest.json` (single mode) or `manifest.json` (matrix mode)

## License

This project is licensed under the [MIT License](LICENSE).

The `sd-scripts/` submodule (kohya-ss/sd-scripts) is licensed under the [Apache License 2.0](sd-scripts/LICENSE.md). It is included as a git submodule and must be initialized with `git submodule update --init --recursive`.
