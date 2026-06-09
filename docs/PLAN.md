# LoRA Matrix Trainer — Plan

## 1. Project Overview

A full-stack Next.js 16 application that provides:
- **Per-model-type tabs** for configuring LoRA training parameters (starting with **Anima**)
- **Single training runs** — start one training job with chosen parameters
- **Matrix training runs** — specify multiple values per parameter and train all permutations
- **Automated evaluation** — use `sd-cli` (stable-diffusion.cpp) to generate comparison images for every trained LoRA
- **Results dashboard** — browse, compare, and inspect all trained LoRAs side-by-side

---

## 2. Supported Model Types (Future-Proof Tab Architecture)

Each model type maps to a kohya-ss training script and has its own tab with type-specific parameters:

| Model Type | Training Script | Notes |
|---|---|---|
| **Anima** (Phase 1) | `anima_train_network.py` | DiT-based, Qwen3 text encoder, Qwen-Image VAE |
| FLUX.1 | `flux_train_network.py` | Phase 2+ |
| SD3 / SD3.5 | `sd3_train_network.py` | Phase 2+ |
| SDXL | `sdxl_train_network.py` | Phase 2+ |
| SD 1.x / 2.x | `train_network.py` | Phase 2+ |
| Hunyuan Image | `hunyuan_image_train_network.py` | Phase 2+ |
| Lumina | `lumina_train_network.py` | Phase 2+ |

---

## 3. Pre-Training Setup Steps

### 3.1. GPU & CUDA Toolkit Installation

- **Detect GPU series** at first launch (`nvidia-smi`):
  - **RTX 50-series (Blackwell)** → Requires **CUDA 13.x** toolkit → PyTorch index: `cu130`
  - **RTX 30/40-series (Ampere/Ada)** → Requires **CUDA 12.x** toolkit → PyTorch index: `cu128`
  - Older GPUs (below RTX 30-series) → **Not supported**
- Verify CUDA install: `nvcc --version` and `nvidia-smi`

### 3.2. Python Environment (uv Only)

**All Python dependency management is handled exclusively by `uv`.** No `pip`, `pip3`, `venv`, or `python -m pip` anywhere.

- Install **uv** if not present: `https://github.com/astral-sh/uv`
- The project uses a `pyproject.toml` with `uv sync` / `uv run` workflow
- GPU detection determines which PyTorch index to activate

**`pyproject.toml` structure:**

```toml
[project]
name = "lora-matrix-trainer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "torchvision",
    "accelerate",
    "transformers",
    "diffusers[torch]",
    "safetensors",
    "bitsandbytes",
    "lion-pytorch",
    "pytorch-optimizer",
    "prodigyopt",
    "prodigy-plus-schedule-free",
    "schedulefree",
    "sentencepiece",
    "toml",
    "rich",
    "numpy",
    "einops",
    "opencv-python",
    "ftfy",
    "huggingface-hub",
    "tensorboard",
    "voluptuous",
    "imagesize",
]

# --- CUDA 12.8 index (RTX 30/40 series) ---
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

# --- CUDA 13.0 index (RTX 50 series) ---
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

# --- Route torch packages to the correct index by platform ---
# The app's setup script writes the active marker based on GPU detection.
# For RTX 30/40 (CUDA 12):
[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
# For RTX 50 (CUDA 13), swap cu128 → cu130 in the above.
```

**Setup workflow (handled by the app's setup script):**

1. Detect GPU via `nvidia-smi` → determine CUDA version (cu128 or cu130)
2. Generate the correct `pyproject.toml` with the matching `[tool.uv.sources]` index routing
3. Run `uv sync` to create the `.venv` and install all dependencies with the correct PyTorch CUDA wheels
4. All subsequent Python execution uses `uv run <script>` — uv automatically uses the managed `.venv`

**Running training scripts:**
```bash
uv run accelerate launch --num_cpu_threads_per_process 1 sd-scripts/anima_train_network.py ...
uv run scripts/matrix_trainer.py ...
uv run scripts/matrix_evaluator.py ...
```

### 3.3. stable-diffusion.cpp Build

- Clone `leejet/stable-diffusion.cpp` (with `--recursive` for submodules)
- Build with CUDA (uses the system CUDA toolkit, must match GPU):
  ```
  cd stable-diffusion.cpp
  mkdir build && cd build
  cmake .. -DSD_CUDA=ON
  cmake --build . --config Release
  ```
- Resulting binary: `bin/Release/sd-cli.exe` (Windows) or `bin/sd-cli` (Linux)
- **Store the binary path** in the app's configuration so Python scripts can invoke it

### 3.4. Base Model Downloads (Anima)

Download the following files from HuggingFace (using `huggingface-cli download`):

| Component | HF Path | Local Destination | Size |
|---|---|---|---|
| **Diffusion Model** | `circlestone-labs/Anima:main` → `split_files/diffusion_models/anima-base-v1.0.safetensors` | `models/anima/diffusion_models/anima-base-v1.0.safetensors` | ~4.18 GB |
| **VAE** | `circlestone-labs/Anima:main` → `split_files/vae/qwen_image_vae.safetensors` | `models/anima/vae/qwen_image_vae.safetensors` | ~254 MB |
| **Text Encoder (Qwen3-0.6B)** | `circlestone-labs/Anima:main` → `split_files/text_encoders/qwen_3_06b_base.safetensors` | `models/anima/text_encoders/qwen_3_06b_base.safetensors` | ~1.2 GB (est.) |

Download commands (hardcoded in the app):
```bash
huggingface-cli download circlestone-labs/Anima split_files/diffusion_models/anima-base-v1.0.safetensors --local-dir models/anima
huggingface-cli download circlestone-labs/Anima split_files/vae/qwen_image_vae.safetensors --local-dir models/anima
huggingface-cli download circlestone-labs/Anima split_files/text_encoders/qwen_3_06b_base.safetensors --local-dir models/anima
```

The app should provide a **"Download Models"** button in the UI that triggers these downloads server-side and shows progress.

---

## 4. Application Architecture

### 4.1. Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16 (App Router), React, TailwindCSS (or shadcn/ui) |
| **Backend API** | Next.js Server Routes (`app/api/...`) |
| **Python Scripts** | Standalone `.py` files invoked via `uv run` |
| **Task Queue** | In-memory job tracker (file-based status) — no external queue needed initially |
| **Storage** | Local filesystem for models, LoRAs, training outputs, and generated images |
| **Process Management** | Node.js `child_process` to launch `uv run <script>`; track PIDs and status |

### 4.2. Directory Structure

```
project-root/
├── app/                          # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx                  # Main dashboard
│   ├── api/
│   │   ├── setup/
│   │   │   └── route.ts          # GPU detection, pyproject.toml generation, uv sync
│   │   ├── models/
│   │   │   └── route.ts          # Download/check model status
│   │   ├── train/
│   │   │   ├── route.ts          # Start single training
│   │   │   └── matrix/route.ts   # Start matrix training
│   │   ├── jobs/
│   │   │   └── route.ts          # List jobs, get job status
│   │   ├── evaluate/
│   │   │   └── route.ts          # Trigger matrix evaluation
│   │   └── results/
│   │       └── route.ts          # Browse results, images
│   └── components/
│       ├── ModelTabs.tsx          # Tab bar (Anima, FLUX, SD3, ...)
│       ├── AnimaTab.tsx           # Anima-specific parameter form
│       ├── MatrixConfig.tsx       # Multi-value parameter inputs
│       ├── JobList.tsx            # Active/completed jobs
│       └── ResultsGrid.tsx        # Side-by-side LoRA comparison
├── scripts/
│   ├── setup_env.py              # GPU detection + pyproject.toml writer
│   ├── matrix_trainer.py          # Matrix permutation trainer
│   ├── matrix_evaluator.py        # sd-cli evaluation runner
│   └── train_single.py            # Single training wrapper
├── sd-scripts/                    # Cloned kohya-ss/sd-scripts
├── stable-diffusion.cpp/          # Cloned and built stable-diffusion.cpp
├── models/
│   └── anima/
│       ├── diffusion_models/
│       ├── vae/
│       └── text_encoders/
├── outputs/                       # All training results
├── pyproject.toml                 # uv-managed Python dependencies (generated by setup)
├── uv.lock                        # uv lockfile (generated by uv sync)
├── package.json                   # Node.js dependencies
└── PLAN.md                        # This file
```

---

## 5. Matrix Trainer Script (`scripts/matrix_trainer.py`)

### 5.1. Purpose

Given a set of parameter ranges, generate all permutations and train a LoRA for each.

### 5.2. Input Format

Parameters accept comma-separated values. Special syntax for percentages:
- `"1,4,8,25%"` means: `1, 4, 8, and 25% of network_dim` — resolved per permutation
- `"1e-4,5e-4,1e-3"` means: literal float values

### 5.3. Permutation Generation

For the example command:
```
--network-dim 1,2,3,4,5,6,7,8
--network-alpha "1,4,8,25%"
--learning-rate "1e-4,5e-4,1e-3"
--batch-size "1,2,4"
--epochs "10,20"
--optimizer "AdamW8Bit,Prodigy"
--learning-rate-scheduler "cosine,constant"
```

Total permutations = 8 × 4 × 3 × 4 × 2 × 2 × 2 = **3,072 training runs**

The script should:
1. Parse all parameter ranges
2. Compute the Cartesian product (`itertools.product`)
3. Resolve special values (e.g., `25%` of each `network_dim`)
4. Create a manifest file listing all permutations
5. Process sequentially (or with configurable parallelism for multi-GPU)

### 5.4. Output Folder Structure

```
outputs/{run-name}/
├── manifest.json                    # All permutations and their statuses
├── anima_network-dim-1_network-alpha-1_learning-rate-1e-4_batch-size-1_epochs-10_optimizer-AdamW8Bit_learning-rate-scheduler-cosine/
│   ├── {LoRA-Name}-000001.safetensors
│   ├── {LoRA-Name}-000002.safetensors
│   ├── ...
│   └── {LoRA-Name}-000010.safetensors   # Final epoch checkpoint
├── anima_network-dim-1_network-alpha-4_.../
└── ...
```

### 5.5. Training Invocation (Per Permutation)

Each permutation invokes `accelerate launch` with `anima_train_network.py`:

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  sd-scripts/anima_train_network.py \
  --pretrained_model_name_or_path "models/anima/diffusion_models/anima-base-v1.0.safetensors" \
  --qwen3 "models/anima/text_encoders/qwen_3_06b_base.safetensors" \
  --vae "models/anima/vae/qwen_image_vae.safetensors" \
  --dataset_config "<generated-toml-path>" \
  --output_dir "<permutation-output-folder>" \
  --output_name "{LoRA-Name}" \
  --save_model_as=safetensors \
  --network_module=networks.lora_anima \
  --network_dim={dim} \
  --network_alpha={alpha} \
  --learning_rate={lr} \
  --train_batch_size={batch-size} \
  --max_train_epochs={epochs} \
  --optimizer_type={optimizer} \
  --lr_scheduler={scheduler} \
  --timestep_sampling=sigmoid \
  --discrete_flow_shift=1.0 \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --save_every_n_epochs=1 \
  --vae_chunk_size=64 \
  --vae_disable_cache
```

### 5.6. Dataset TOML Generation

The script auto-generates a `.toml` dataset config from the user's training images path:

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = 512
batch_size = {batch-size}

  [[datasets.subsets]]
  image_dir = '{training-images-path}'
  num_repeats = {calculated-from-epochs}
```

### 5.7. Job Tracking

- Each matrix run gets a unique ID (timestamp + random suffix)
- `manifest.json` tracks status per permutation: `pending`, `running`, `completed`, `failed`
- Real-time updates via polling the API (`/api/jobs/[id]`)

---

## 6. Matrix Evaluator Script (`scripts/matrix_evaluator.py`)

### 6.1. Purpose

Given a results folder, evaluate every trained LoRA by generating an image with `sd-cli` using a consistent prompt and seed.

### 6.2. Steps

1. **Scan the results folder** for all permutation subdirectories
2. **Find all `.safetensors` LoRA files** (pick the highest-numbered epoch checkpoint, e.g., `-000010.safetensors`)
3. **Extract tags from training data**: Read caption files (`.txt`) from the original training images to collect all tags
4. **Generate a single random prompt**: Combine a random subset of tags into one prompt
5. **Run sd-cli for each LoRA**:
   ```bash
   sd-cli \
     --diffusion-model "models/anima/diffusion_models/anima-base-v1.0.safetensors" \
     --vae "models/anima/vae/qwen_image_vae.safetensors" \
     --llm "models/anima/text_encoders/qwen_3_06b_base.safetensors" \
     --lora-model-dir "<permutation-folder>" \
     -p "<prompt><lora:{lora-filename-without-ext}:1>" \
     --cfg-scale 6.0 \
     --sampling-method euler \
     -s {fixed-seed} \
     --steps 20 \
     --diffusion-fa \
     --offload-to-cpu \
     -o "<permutation-folder>/<lora-name>.png"
   ```
6. **Output**: Each permutation folder gets an evaluation PNG alongside the LoRA files

### 6.3. Evaluation Metadata

Write `evaluation.json` in the root results folder:
```json
{
  "prompt": "the generated prompt used for all evaluations",
  "seed": 42,
  "cfg_scale": 6.0,
  "steps": 20,
  "results": [
    {
      "permutation": "network-dim-1_network-alpha-1_...",
      "lora_file": "LoRA-Name-000010.safetensors",
      "image_file": "LoRA-Name-000010.png",
      "status": "success" | "failed",
      "inference_time_ms": 12345
    }
  ]
}
```

---

## 7. UI Design (Anima Tab — Phase 1)

### 7.1. Parameter Form Fields

| Parameter | UI Widget | Default | Notes |
|---|---|---|---|
| **Network Dim** | Number input (or multi-value for matrix) | 8 | Single or comma-separated |
| **Network Alpha** | Number input (supports `%` suffix) | 1 | Single or comma-separated |
| **Learning Rate** | Number input (scientific notation) | 1e-4 | Single or comma-separated |
| **Batch Size** | Number input | 1 | Single or comma-separated |
| **Epochs** | Number input | 10 | Single or comma-separated |
| **Optimizer** | Dropdown | AdamW8Bit | Options: AdamW8Bit, AdamW, Prodigy, Lion, Adafactor |
| **LR Scheduler** | Dropdown | cosine | Options: constant, cosine, linear, constant_with_warmup, cosine_with_restarts |
| **Training Images** | Folder picker / path input | — | Path to images + .txt captions |
| **LoRA Name** | Text input | — | Name for the output LoRA |
| **Mixed Precision** | Dropdown | bf16 | fp16, bf16, no |
| **Timestep Sampling** | Dropdown | sigmoid | sigma, uniform, sigmoid, shift, flux_shift |
| **Gradient Checkpointing** | Toggle | ON | Memory optimization |
| **Cache Latents** | Toggle | ON | Pre-compute VAE outputs |
| **Cache Text Encoder** | Toggle | ON | Pre-compute TE outputs |

### 7.2. Mode Toggle

A prominent toggle: **Single Run** ↔ **Matrix Run**
- In Single mode: each parameter is a single value
- In Matrix mode: each parameter accepts comma-separated values (or a tag-style multi-input)
- Matrix mode shows a **permutation count** (e.g., "3,072 combinations")

### 7.3. Jobs Panel

- List of active and recent jobs
- Each job shows: name, status (running/completed/failed), progress (%), permutation count
- Click to expand: individual permutation statuses

### 7.4. Results Viewer

- Grid of evaluation images, one per permutation
- Each card shows: parameter values, LoRA file link, evaluation image
- Sort/filter by parameter values
- Side-by-side comparison mode (select 2+ images)

---

## 8. Implementation Phases

### Phase 0: Foundation
- [ ] Set up Next.js 16 project with App Router
- [ ] Create `pyproject.toml` template with uv index configuration for both cu128 and cu130
- [ ] Write `scripts/setup_env.py` — GPU detection via `nvidia-smi`, writes correct `[tool.uv.sources]` routing, runs `uv sync`
- [ ] Implement `/api/setup` endpoint — triggers setup_env.py, reports GPU/CUDA status
- [ ] Clone kohya-ss/sd-scripts as a project subdirectory
- [ ] Clone and build stable-diffusion.cpp with CUDA
- [ ] Add setup wizard UI (detect GPU → configure uv → verify)

### Phase 1: Model Downloads
- [ ] Implement server-side model download API (`/api/models`)
- [ ] Hardcode Anima HuggingFace paths
- [ ] Implement download progress tracking
- [ ] Add "Download Models" UI button with progress indicator
- [ ] Verify downloaded files are valid (file size, safetensors header check)

### Phase 2: Single Training
- [ ] Write `scripts/train_single.py` wrapper
- [ ] Implement `/api/train` endpoint
- [ ] Build Anima parameter form UI
- [ ] Implement job tracking (in-memory + file-based manifest)
- [ ] Add training log streaming to UI
- [ ] Test end-to-end: configure → train → verify output

### Phase 3: Matrix Training
- [ ] Write `scripts/matrix_trainer.py`
- [ ] Implement permutation generation and `%` value resolution
- [ ] Implement `/api/train/matrix` endpoint
- [ ] Add matrix mode toggle to UI
- [ ] Add permutation count display
- [ ] Implement real-time progress tracking per permutation
- [ ] Add pause/resume/cancel functionality

### Phase 4: Evaluation
- [ ] Write `scripts/matrix_evaluator.py`
- [ ] Implement tag extraction from training captions
- [ ] Implement random prompt generation
- [ ] Implement sd-cli invocation for each LoRA
- [ ] Implement `/api/evaluate` endpoint
- [ ] Write evaluation.json metadata
- [ ] Add "Evaluate All" button to UI

### Phase 5: Results Dashboard
- [ ] Build results grid component
- [ ] Implement image browsing from result folders
- [ ] Add parameter-based filtering/sorting
- [ ] Implement side-by-side comparison view
- [ ] Add LoRA file download links
- [ ] Implement manifest.json parsing for result display

### Phase 6: Polish & Robustness
- [ ] Error handling and user-friendly error messages
- [ ] Training log viewer (scrollable, searchable)
- [ ] GPU VRAM monitoring during training
- [ ] Config file save/load (save parameter presets)
- [ ] Documentation and README

---

## 9. Key Dependencies

### Python (managed exclusively by uv)
- torch (with CUDA, via `[[tool.uv.index]]` + `[tool.uv.sources]` routing) — deep learning framework
- torchvision (same CUDA index routing)
- accelerate — required by training scripts
- transformers — text encoder support
- diffusers — model utilities
- safetensors — model file format
- bitsandbytes — 8-bit optimizers
- lion-pytorch — Lion optimizer
- pytorch-optimizer — extended optimizer collection
- prodigyopt, prodigy-plus-schedule-free — Prodigy optimizer
- schedulefree — schedule-free optimizers
- sentencepiece — T5 tokenizer
- toml — dataset config parsing
- rich — training output formatting
- numpy, einops — tensor utilities
- opencv-python — image processing
- ftfy — text normalization
- huggingface-hub — model downloads
- tensorboard — training logs
- voluptuous — config validation
- imagesize — image dimension detection

### Node.js (managed by bun)
- next (v16 or latest canary)
- react, react-dom
- tailwindcss (or shadcn/ui components)
- framer-motion (animations)
- lucide-react (icons)

### System-Level
- **CUDA Toolkit** (12.x for RTX 30/40, 13.x for RTX 50)
- **CMake** (for building stable-diffusion.cpp)
- **MSVC** (Windows C++ compiler for stable-diffusion.cpp) / **GCC** (Linux)
- **huggingface-cli** (for model downloads — installed via `uv pip install huggingface-hub[inference]` or as a standalone tool)
- **Git with submodules** (for stable-diffusion.cpp clone)

---

## 10. Known Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CUDA version mismatch between PyTorch and sd.cpp | Training/inference fails | Detect GPU and route PyTorch to correct uv index; build sd.cpp from source against system CUDA |
| Large permutation counts (3000+) | Very long total training time | Add pause/resume; allow pruning permutations; show live progress |
| VRAM exhaustion during training | Training crashes | Auto-suggest `--blocks_to_swap` or `--unsloth_offload_checkpointing` based on GPU VRAM |
| sd.cpp Anima LoRA compatibility | Evaluation may not work | Test early; fallback to Python-based inference (kohya-ss `anima_minimal_inference.py`) if needed |
| Model download failures (large files, network issues) | Blocked setup | Retry logic; resume support; progress persistence |
| safetensors format differences between training and inference | LoRA not loading correctly | Test with known-good LoRA first; implement format conversion if needed |
| uv index routing on first sync fails | Dependencies not installed | Validate `pyproject.toml` before `uv sync`; provide clear error with CUDA version guidance |

---

## 11. Open Questions

1. **sd.cpp Anima LoRA support**: Does `sd-cli` correctly load and apply LoRA files trained with `anima_train_network.py`? Need to verify — the LoRA key naming may differ between sd-scripts format and sd.cpp expectations. Fallback: use kohya-ss's `anima_minimal_inference.py` for evaluation.
2. **Parallel training**: Can we train multiple permutations simultaneously on multi-GPU systems?
3. **Evaluation metrics**: Beyond visual comparison, should we implement automated quality metrics (e.g., CLIP score, FID)?
4. **Dataset format**: Should the app auto-generate captions if none exist (using a captioning model)?
5. **Persistence**: Should job state survive server restarts? (File-based manifest covers this.)

---

## 12. Next Steps

1. **Approve this plan** and decide on any changes to scope or architecture
2. **Set up the project skeleton** (Next.js + pyproject.toml with uv CUDA index routing)
3. **Verify GPU/CUDA environment** on the target machine
4. **Test a manual training run** end-to-end before automating
5. **Test sd.cpp with an Anima LoRA** to confirm compatibility (or plan fallback to Python inference)
