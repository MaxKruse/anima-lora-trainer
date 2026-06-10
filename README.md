# LoRA Matrix Trainer

Train Anima LoRA models with matrix-style hyperparameter sweeps and automatic test-prompt inference.

## Quick Start

```bash
# Install Python dependencies
uv sync

# Start the web UI
bun dev
```

Open [http://localhost:3000](http://localhost:3000) to configure and launch training.

## Training Modes

### Single Run
Train one LoRA with fixed hyperparameters. The web UI validates params and launches training. After training completes, a test-prompt inference generates sample images in `sample_images/`.

### Matrix Run
Define ranges for any parameter (network dim, learning rate, epochs, etc.) and train every permutation. A single test prompt is generated from your training data tags and reused across all runs — sample images are directly comparable.

## Ad-Hoc Batch Inference

Run test-prompt inference on existing LoRA files without retraining:

```bash
# Inference with a manual prompt:
uv run python scripts/infer_batch.py --dir output/job-123 --prompt "masterpiece, 1girl, solo, red hair"

# Auto-generate prompt from training data captions:
uv run python scripts/infer_batch.py --dir output/ --training-images datasets/mari_setogaya/img

# Custom settings:
uv run python scripts/infer_batch.py --dir output/ --prompt "masterpiece, 1girl" --seed 1234 --steps 40
```

Scans the directory recursively for `.safetensors` files, runs `sd-cli` inference on each, and writes output to `sample_images/` alongside each LoRA.

### Options

| Flag | Description |
|------|-------------|
| `--dir, -d` | Directory to scan for LoRAs (required) |
| `--prompt, -p` | Test prompt (comma-separated tags) |
| `--training-images, -t` | Training data dir (auto-generates prompt from .txt captions) |
| `--seed, -s` | Random seed (default: 42) |
| `--steps` | Inference steps (default: 30) |
| `--cfg-scale` | CFG scale (default: 4.0) |
| `--diffusion-model` | Path to diffusion model |
| `--vae` | Path to VAE model |
| `--llm` | Path to text encoder model |
| `--dry-run` | List LoRAs without running inference |

## Inference Settings

Test-prompt inference uses Anima-optimized defaults:

| Setting | Value | Notes |
|---------|-------|-------|
| Sampler | `euler` | Default for Flow models |
| Scheduler | `simple` | Flow matching schedule |
| CFG Scale | `4.0` | Sweet spot 3–5 for Anima (lower than SD's 7) |
| Steps | `30` | Typical range 25–35 |
| Negative Prompt | `worst quality, low quality, blurry, bad anatomy, deformed hands` | Recommended baseline |

## Project Structure

```
scripts/
  train_single.py       Single training run (dataset TOML → accelerate → inference)
  matrix_trainer.py     Matrix training (permute → train each → inference)
  infer_batch.py        Ad-hoc batch inference on existing LoRAs
  prompt_generator.py   Generate test prompts from training tags
  tag_extractor.py      Extract tags from .txt caption files
  sdcli_builder.py      Build sd-cli inference commands
  command_builder.py    Build accelerate/kooya-ss training commands
  dataset_toml.py       Generate kohya-ss dataset config
```
