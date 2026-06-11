# LoRA Matrix Trainer

Train Anima character LoRA models via CLI — single runs or matrix hyperparameter sweeps.

## Quick Start

```bash
# Install Python dependencies
uv sync

# 1. Validate your dataset (mandatory — creates output dir)
uv run python scripts/train.py --validate --dataset datasets/froot/img

# 2. Train a single LoRA (uses proven defaults)
uv run python scripts/train.py --mode single --dataset datasets/froot/img --name Froot-Anima

# 3. Matrix sweep (all permutations)
uv run python scripts/train.py --mode matrix --dataset datasets/froot/img --name Froot --network-dim 16,20,32 --alpha 1,20
```

**Validation is mandatory.** Training will refuse to start until you've run `--validate` on the dataset. Validation:
- Creates the `out/` directory for training results
- Checks image counts and caption coverage
- Verifies per-folder image limits (warns if > 25 per subfolder)
- Writes a `.validation.json` marker file

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
  base/            ← base character images (14 images)
  outfit1/         ← outfit variation 1 (3 images)
  outfit2/         ← outfit variation 2 (2 images)
```

Each subdirectory gets its own `[[datasets.subsets]]` entry in the generated `dataset.toml`, so kohya-ss trains on all of them. The `--validate` flag shows the folder breakdown.

**Recommended:** 12–20 total images (across all folders) spanning different poses, outfits, and aspect ratios.

### Validate Before Training (Mandatory)

```bash
uv run python scripts/train.py --validate --dataset datasets/froot/img
```

**Must be run before any training.** Creates the output directory and writes a validation marker.

Checks:
- Image count per folder (warns if > 25 per subfolder)
- Total image count (recommends 12–20, minimum 10)
- Caption coverage (each image should have a matching .txt)
- Calculates recommended `num_repeats` for your image count

Warnings are non-blocking — training proceeds even with warnings. Only hard errors (missing directory, no images) prevent training.

## Training Modes

### Single Run

Train one LoRA with fixed parameters. All flags default to values from proven training runs (froot: 44 images, mari_setogaya: 14 images).

```bash
uv run python scripts/train.py --mode single \
  --dataset datasets/froot/img \
  --name Froot-Anima \
  --lr 0.0002 \
  --bs 4 \
  --network-dim 20 \
  --alpha 1
```

### Matrix Run

Specify comma-separated values for any parameter — all permutations are trained sequentially with a shared test prompt for comparison.

```bash
uv run python scripts/train.py --mode matrix \
  --dataset datasets/froot/img \
  --name Froot \
  --network-dim 16,20,32 \
  --alpha 1,16,20 \
  --lr 0.0001,0.0002
```

This generates 3 × 3 × 2 = **18 training runs**.

**Resume** an interrupted matrix run:
```bash
uv run python scripts/train.py --mode matrix --resume ...
```

**Cancel** a running job:
```bash
# Single mode: touch the cancel file in jobs/
echo > jobs/job-<timestamp>-<id>.cancel

# Matrix mode: touch cancel in the output dir
echo > datasets/<name>/out/job-<timestamp>-<id>/cancel
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
| `--validate`, `-v` | Validate dataset and exit |

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
| `--optimizer` | AdamW8Bit | Optimizer type |
| `--scheduler`, `-s` | cosine | LR scheduler |
| `--resolution` | 1024 | Image resolution (768–1024) |
| `--repeats`, `-r` | auto | Override num_repeats (auto-calculated from image count) |
| `--mixed-precision` | bf16 | fp16 / bf16 / no |
| `--timestep-sampling` | sigmoid | sigma / uniform / sigmoid / shift / flux_shift |
| `--caption-dropout` | 0.05 | Caption tag dropout rate (0.0–1.0) |
| `--keep-tokens` | 1 | Keep first N tokens from caption shuffle |
| `--no-gradient-checkpointing` | off | Disable gradient checkpointing |
| `--no-cache-latents` | off | Disable latent caching |
| `--cache-text-encoder` | off | Enable text encoder caching |

### Matrix

| Flag | Description |
|------|-------------|
| `--resume` | Resume from existing manifest (skip completed permutations) |

## Default Parameters (Proven Recipe)

These defaults come from successful training runs on real character datasets:

| Parameter | Value | Why |
|-----------|-------|-----|
| `network_dim` | 20 | Sweet spot for character detail (16–32 range) |
| `network_alpha` | 1 | Effective scale 0.05 — lets you prompt detail in/out |
| `learning_rate` | 0.0002 | Aggressive but stable with cosine decay |
| `batch_size` | 4 | Balanced gradient signal vs VRAM (~17GB) |
| `max_steps` | 800 | Enough for convergence, checkpoints every 80 steps |
| `optimizer` | AdamW8Bit | Memory-efficient, no convergence sacrifice |
| `scheduler` | cosine | Smooth decay from peak LR to near-zero |
| `mixed_precision` | bf16 | Required for Anima/Qwen3 |
| `timestep_sampling` | sigmoid | Focuses on mid-to-high noise levels |
| `caption_tag_dropout` | 0.05 | 5% random tag drops → better generalization |
| `keep_tokens` | 1 | Preserves trigger word from caption shuffle |

## Output Structure

```
datasets/froot/out/job-1781162746594-khr542/
  dataset.toml              ← generated dataset config
  training.log              ← training log
  job_manifest.json         ← progress/status
  Froot-Anima-step00000080.safetensors
  Froot-Anima-step00000160.safetensors
  ...
  Froot-Anima-step00000800.safetensors
  Froot-Anima.safetensors   ← final checkpoint
  training-data.zip         ← backup of training data
```

Matrix runs create subdirectories per permutation under the job folder.

## Project Structure

```
scripts/
  train.py                  ← unified CLI (single + matrix modes)
  command_builder.py        ← builds accelerate/kohya-ss commands
  dataset_toml.py           ← generates kohya-ss dataset config
  train_single.py           ← legacy single training (kept for compat)
  matrix_trainer.py         ← legacy matrix training (kept for compat)
  infer_batch.py            ← ad-hoc batch inference on existing LoRAs
  prompt_generator.py       ← generate test prompts from tags
  tag_extractor.py          ← extract tags from .txt captions
  sdcli_builder.py          ← build sd-cli inference commands
sd-scripts/                 ← kohya-ss/sd-scripts (training engine)
models/                     ← downloaded base models
datasets/                   ← training datasets
```
