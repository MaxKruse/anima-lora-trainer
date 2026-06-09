# LoRA Matrix Trainer — Task Breakdown

Each task follows TDD: write the failing test first, then implement minimal code to pass. Tasks are ordered so earlier tasks unblock later ones.

**Test summary:** 149 tests passing (71 TypeScript via Vitest + 78 Python via pytest)

---

## Phase 0: Foundation ✅ COMPLETE

### Task 0.1 ✅ — GPU detection module (Python)

**What:** `scripts/setup_env.py` — parse `nvidia-smi` output and classify GPU into CUDA tier.

**Tests:** `tests/test_gpu_detection.py` — **6 tests** ✅
- [x] Given nvidia-smi output containing "RTX 50", return `{cuda: "cu130", series: "blackwell"}`
- [x] Given nvidia-smi output containing "RTX 40", return `{cuda: "cu128", series: "ada"}`
- [x] Given nvidia-smi output containing "RTX 30", return `{cuda: "cu128", series: "ampere"}`
- [x] Given nvidia-smi output with no recognized GPU, raise `UnsupportedGpuError`
- [x] Given empty/missing nvidia-smi output, raise `NvidiaSmiError`
- [x] Given None output, raise `NvidiaSmiError`

### Task 0.2 ✅ — pyproject.toml generator (Python)

**What:** Function that writes a `pyproject.toml` with correct `[tool.uv.sources]` routing for the detected CUDA tier.

**Tests:** `tests/test_pyproject_generator.py` — **5 tests** ✅
- [x] Given cu128, generated toml routes torch and torchvision to `pytorch-cu128` index
- [x] Given cu130, generated toml routes torch and torchvision to `pytorch-cu130` index
- [x] Generated toml contains all required dependencies from the spec
- [x] Generated toml is valid TOML (parse round-trip)
- [x] Both CUDA indexes defined in generated toml

### Task 0.3 ✅ — Setup API endpoint

**What:** `app/api/setup/route.ts` — triggers GPU detection and pyproject generation.

**Tests:** `app/api/setup/route.test.ts` — **8 tests** ✅
- [x] POST returns `{ gpu, cuda, status: "ok" }` when setup succeeds
- [x] POST returns 500 with error detail when nvidia-smi fails
- [x] Response includes the resolved CUDA version string
- [x] `parseGpuInfo` returns cu130 for RTX 50 series
- [x] `parseGpuInfo` returns cu128 for RTX 40 series
- [x] `parseGpuInfo` returns cu128 for RTX 30 series
- [x] `parseGpuInfo` returns null for unsupported GPU
- [x] `parseGpuInfo` returns null for empty input

### Task 0.4 ✅ — Setup wizard UI

**What:** `app/components/SetupWizard.tsx` — GPU detection button, CUDA version result, and environment status.

**Tests:** `app/components/__tests__/SetupWizard.test.tsx` — **4 tests** ✅
- [x] Renders "Detect GPU" button initially
- [x] After successful detection, displays GPU name and CUDA version
- [x] Shows error message when detection fails
- [x] Shows "Environment ready" when pyproject.toml exists

---

## Phase 1: Model Downloads ✅ COMPLETE

### Task 1.1 ✅ — Model manifest module (TypeScript)

**What:** `app/lib/model-manifest.ts` — Anima model download specs (HF path, local destination, expected size).

**Tests:** `app/lib/__tests__/model-manifest.test.ts` — **7 tests** ✅
- [x] Returns 3 entries for Anima (diffusion model, VAE, text encoder)
- [x] Each entry has `name`, `hfPath`, `localPath`, and `expectedSizeBytes`
- [x] Local paths are under `models/anima/`
- [x] Includes diffusion model entry
- [x] Includes VAE entry
- [x] Includes text encoder entry
- [x] Throws for unknown model type

### Task 1.2 ✅ — Model download service (TypeScript)

**What:** `app/lib/model-downloader.ts` — invokes `huggingface-cli download` and streams progress.

**Tests:** `app/lib/__tests__/model-downloader.test.ts` — **4 tests** ✅
- [x] Calls huggingface-cli with correct args for each model entry
- [x] Reports progress as download completes
- [x] Retries on transient network error (up to 3 attempts)
- [x] Throws after 3 failed attempts

### Task 1.3 ✅ — Model verification (Python)

**What:** `scripts/model_verify.py` — checks a downloaded `.safetensors` file has a valid header.

**Tests:** `tests/test_model_verify.py` — **7 tests** ✅
- [x] Given a valid safetensors file path, return `True`
- [x] Given a valid safetensors file with tensor data, return `True`
- [x] Given a truncated file, return `False`
- [x] Given an empty file, return `False`
- [x] Given a non-existent file, raise `FileNotFoundError`
- [x] Given invalid header length, return `False`
- [x] Given invalid JSON header, return `False`

### Task 1.4 ✅ — Models API endpoint

**What:** `app/api/models/route.ts` — download, check status, verify.

**Tests:** `app/api/models/route.test.ts` — **5 tests** ✅
- [x] GET returns status of all models (downloaded / pending / verifying)
- [x] GET shows downloaded status for existing models
- [x] POST triggers download of a specific model
- [x] Returns 409 if model already downloaded
- [x] Returns 422 if verification fails after download

### Task 1.5 ✅ — Download UI button with progress

**What:** `app/components/ModelDownloader.tsx` — "Download Models" button with per-model progress bars.

**Tests:** `app/components/__tests__/ModelDownloader.test.tsx` — **4 tests** ✅
- [x] Renders button for each pending model
- [x] Shows progress bar during download
- [x] Shows checkmark when download completes
- [x] Shows error message when download fails

---

## Phase 2: Single Training ✅ COMPLETE (9/9)

### Task 2.1 ✅ — Training parameter schema (TypeScript)

**What:** `app/lib/training-schema.ts` — Validation schema for Anima training parameters (zod).

**Tests:** `app/lib/__tests__/training-schema.test.ts` — **14 tests** ✅
- [x] Accepts valid single-run parameter set
- [x] Rejects missing required field: network_dim
- [x] Rejects missing required field: learning_rate
- [x] Rejects missing required field: epochs
- [x] Rejects negative network_dim
- [x] Rejects zero epochs
- [x] Rejects negative learning_rate
- [x] Rejects negative batch_size
- [x] Accepts all optimizer types from spec (AdamW8Bit, AdamW, Prodigy, Lion, Adafactor)
- [x] Rejects invalid optimizer
- [x] Accepts all scheduler types from spec (constant, cosine, linear, constant_with_warmup, cosine_with_restarts)
- [x] Rejects invalid scheduler
- [x] Accepts all mixed precision options (fp16, bf16, no)
- [x] Accepts all timestep sampling options (sigma, uniform, sigmoid, shift, flux_shift)

### Task 2.2 ✅ — Dataset TOML generator (Python)

**What:** `scripts/dataset_toml.py` — generates a `.toml` dataset config from user-provided image path and batch size.

**Tests:** `tests/test_dataset_toml.py` — **7 tests** ✅
- [x] Given image_dir and batch_size, produces valid TOML with correct structure
- [x] `num_repeats` is calculated as `ceil(desired_steps / num_images)`
- [x] `num_repeats` rounds up with remainder (ceil)
- [x] Sets `caption_extension = '.txt'`
- [x] Sets `shuffle_caption = true`
- [x] Writes to specified output path
- [x] Supports custom resolution

### Task 2.3 ✅ — Training command builder (Python)

**What:** `scripts/command_builder.py` — assembles the full `accelerate launch` command from parameters.

**Tests:** `tests/test_command_builder.py` — **12 tests** ✅
- [x] Given Anima params, produces command with all required flags from spec
- [x] Includes correct model paths for diffusion model, VAE, text encoder
- [x] Sets `--network_module=networks.lora_anima`
- [x] Includes `--mixed_precision=bf16`
- [x] Includes `--gradient_checkpointing`
- [x] Includes `--cache_latents`
- [x] Includes `--cache_text_encoder_outputs`
- [x] Includes `--timestep_sampling=sigmoid`
- [x] Includes `--discrete_flow_shift=1.0`
- [x] Includes `--vae_chunk_size=64` and `--vae_disable_cache`
- [x] Includes `--save_every_n_epochs=1`
- [x] Includes correct network_dim, alpha, lr, batch_size, epochs, optimizer, scheduler, output_name

### Task 2.4 ✅ — Single training script

**What:** `scripts/train_single.py` — parses args, generates dataset TOML, builds command, launches training.

**Tests:** `tests/test_train_single.py` — **5 tests** ✅
- [x] Creates output directory before launching
- [x] Writes initial job manifest with status `running`
- [x] Updates manifest to `completed` on exit code 0
- [x] Updates manifest to `failed` on non-zero exit code
- [x] Produces correct command (via subprocess.run call verification)

### Task 2.5 ✅ — Train API endpoint

**What:** `app/api/train/route.ts` — validates params, launches `uv run scripts/train_single.py`.

**Tests:** `app/api/train/__tests__/route.test.ts` — **4 tests** ✅
- [x] POST with valid params returns `{ jobId, status: "started" }`
- [x] POST with invalid params returns 400 with validation errors
- [x] POST when another job is running returns 409
- [x] Job ID is unique (timestamp + random suffix)

### Task 2.6 ✅ — Anima parameter form UI

**What:** `app/components/AnimaTab.tsx` — form with all parameter fields from spec.

**Tests:** `app/components/__tests__/AnimaTab.test.tsx` — **4 tests** ✅
- [x] Renders all fields from spec (network dim, alpha, lr, batch size, epochs, optimizer, scheduler, training images, lora name, mixed precision, timestep sampling, gradient checkpointing, cache latents, cache text encoder)
- [x] Each field has correct default value
- [x] Submitting fires callback with correct param object
- [x] Validates required fields before submit

### Task 2.7 ✅ — Job tracking store (TypeScript)

**What:** In-memory + file-based job state tracker.

**Tests:** `app/lib/__tests__/job-store.test.ts` — **5 tests** ✅
- [x] `createJob(params)` returns unique job ID and stores job
- [x] `getJob(id)` returns job with current status
- [x] `listJobs()` returns all jobs
- [x] Job state persists to file (survives process restart)
- [x] Loading from file restores all jobs

### Task 2.8 ✅ — Jobs API endpoint

**What:** `app/api/jobs/route.ts` — list jobs, get individual job status.

**Tests:** `app/api/jobs/__tests__/route.test.ts` — **3 tests** ✅
- [x] GET returns array of all jobs
- [x] GET with query param `?id=X` returns single job
- [x] Returns empty array when no jobs exist

### Task 2.9 ✅ — Job list UI

**What:** `app/components/JobList.tsx` — displays active and recent jobs.

**Tests:** `app/components/__tests__/JobList.test.tsx` — **3 tests** ✅
- [x] Renders job cards with name, status, progress
- [x] Shows "running", "completed", "failed" status labels
- [x] Expandable to show individual permutation statuses (for matrix jobs)

---

## Phase 3: Matrix Training ✅ COMPLETE (7/7)

### Task 3.1 ✅ — Parameter range parser (Python)

**What:** `scripts/param_parser.py` — parse comma-separated parameter values, including `%` suffix resolution.

**Tests:** `tests/test_param_parser.py` — **10 tests** ✅
- [x] `"1,2,3"` → `[1, 2, 3]` (integers)
- [x] `"1e-4,5e-4,1e-3"` → `[1e-4, 5e-4, 1e-3]` (floats)
- [x] `"AdamW8Bit,Prodigy"` → `["AdamW8Bit", "Prodigy"]` (strings)
- [x] `"1,4,8,25%"` → `[1, 4, 8, "25%"]` (preserves `%` marker for later resolution)
- [x] Empty string raises `ValueError`
- [x] Whitespace-only string raises `ValueError`
- [x] Single value returns list with one element
- [x] Mixed int and float values
- [x] Whitespace around values is trimmed
- [x] Negative numbers are parsed correctly

### Task 3.2 ✅ — Permutation generator (Python)

**What:** `scripts/permutation_generator.py` — compute Cartesian product of parameter ranges and resolve `%` values.

**Tests:** `tests/test_permutation_generator.py` — **7 tests** ✅
- [x] Given `{dim: [1,2], alpha: [1,4]}`, produces 2×2 = 4 permutations
- [x] Given `{dim: [1,2,3], alpha: [1,4], lr: [1e-4]}`, produces 3×2×1 = 6 permutations
- [x] `25%` alpha resolves to `dim * 0.25` for each permutation's dim value
- [x] Each permutation is a flat dict of `{param_name: resolved_value}`
- [x] Large input (8×4×3×4×2×2×2) produces exactly 3,072 permutations
- [x] Single param with single value produces one permutation
- [x] Percent reference to non-numeric param raises `ValueError`

### Task 3.3 ✅ — Permutation folder namer (Python)

**What:** `scripts/permutation_namer.py` — generate deterministic folder names from permutation params.

**Tests:** `tests/test_permutation_namer.py` — **8 tests** ✅
- [x] Params sorted alphabetically: learning-rate < network-alpha < network-dim
- [x] Float values use compact scientific notation (`1e-4`)
- [x] Params sorted alphabetically for deterministic naming
- [x] String param values like optimizer are included
- [x] Model prefix (anima) is included in folder name
- [x] Custom prefix can be specified
- [x] Same params always produce the same folder name (deterministic)
- [x] Dict key order doesn't matter (sorted alphabetically)

### Task 3.4 ✅ — Manifest writer (Python)

**What:** `scripts/manifest_writer.py` — create and update `manifest.json` tracking all permutations and statuses.

**Tests:** `tests/test_manifest.py` — **6 tests** ✅
- [x] Initial manifest has all permutations with status `pending`
- [x] Updating one permutation to `running` persists correctly
- [x] Updating to `completed` stores output file paths
- [x] Updating to `failed` stores error message
- [x] Manifest survives re-read (JSON round-trip)
- [x] Manifest is actually written to disk as valid JSON

### Task 3.5 ✅ — Matrix trainer script

**What:** `scripts/matrix_trainer.py` — parse args, generate permutations, iterate and train each.

**Tests:** `tests/test_matrix_trainer.py` — **5 tests** ✅
- [x] Creates output directory and manifest before training loop
- [x] Processes permutations sequentially (one at a time)
- [x] Updates manifest status for each permutation as it completes
- [x] Stops on `cancel` signal file presence
- [x] Supports `--resume` to skip already-completed permutations

### Task 3.6 ✅ — Matrix train API endpoint

**What:** `app/api/train/matrix/route.ts` — validates matrix params, launches matrix trainer.

**Tests:** `app/api/train/matrix/__tests__/route.test.ts` — **3 tests** ✅
- [x] POST with valid matrix params returns `{ jobId, permutationCount, status: "started" }`
- [x] Rejects params that would produce 0 permutations
- [x] Returns 400 if any parameter range is empty

### Task 3.7 ✅ — Matrix mode toggle UI

**What:** `app/components/MatrixToggle.tsx` — toggle between Single Run and Matrix Run modes.

**Tests:** `app/components/__tests__/MatrixToggle.test.tsx` — **3 tests** ✅
- [x] Default mode is Single Run
- [x] Toggling to Matrix Run fires onChange callback
- [x] Shows permutation count when in Matrix mode

---

## Phase 4: Evaluation

### Task 4.1 ⬜ — Tag extractor (Python)

**What:** Read `.txt` caption files from training image directory and collect unique tags.

**Test first:**
- `tests/test_tag_extractor.py`
  - Given directory with `.txt` files, returns list of all unique tags
  - Splits multi-tag captions (comma-separated or space-separated)
  - Ignores empty caption files
  - Handles missing captions gracefully (no crash)

### Task 4.2 ⬜ — Prompt generator (Python)

**What:** Combine random subset of tags into a single evaluation prompt.

**Test first:**
- `tests/test_prompt_generator.py`
  - Given 20 tags, produces prompt with random subset (5-10 tags)
  - Same seed produces same prompt (deterministic)
  - Different seeds produce different prompts
  - Returns empty string if no tags available

### Task 4.3 ⬜ — LoRA file finder (Python)

**What:** Scan a permutation folder for `.safetensors` files and pick the highest epoch checkpoint.

**Test first:**
- `tests/test_lora_finder.py`
  - Given folder with `{name}-000001.safetensors` through `{name}-000010.safetensors`, returns the `-000010` file
  - Given folder with only one checkpoint, returns it
  - Given empty folder, raises `NoLoraFoundError`
  - Ignores non-safetensors files

### Task 4.4 ⬜ — sd-cli command builder (Python)

**What:** Assemble the `sd-cli` command for evaluating a single LoRA.

**Test first:**
- `tests/test_sdcli_builder.py`
  - Produces command with correct model paths (diffusion, VAE, LLM)
  - Includes `--lora-model-dir` pointing to permutation folder
  - Includes prompt with `<lora:{filename}:1>` syntax
  - Sets `--cfg-scale 6.0`, `--sampling-method euler`, `--steps 20`
  - Sets `--diffusion-fa`, `--offload-to-cpu`
  - Sets fixed seed via `-s {seed}`
  - Sets output path via `-o`

### Task 4.5 ⬜ — Matrix evaluator script

**What:** `scripts/matrix_evaluator.py` — scan results, find LoRAs, run sd-cli, write evaluation.json.

**Test first:**
- `tests/test_matrix_evaluator.py`
  - Scans results folder and finds all permutation subdirectories
  - Picks highest-epoch LoRA for each permutation
  - Runs sd-cli for each LoRA with same prompt and seed
  - Writes `evaluation.json` with correct structure
  - Records `inference_time_ms` per result
  - Records `status: "failed"` for sd-cli errors without stopping entire run

### Task 4.6 ⬜ — Evaluate API endpoint

**What:** `app/api/evaluate/route.ts` — triggers matrix evaluator for a given run.

**Test first:**
- `app/api/evaluate/__tests__/route.test.ts`
  - POST with valid run ID starts evaluation
  - Returns 404 for non-existent run ID
  - Returns 409 if evaluation already running for this run
  - Returns evaluation results when complete

### Task 4.7 ⬜ — "Evaluate All" button UI

**What:** Button to trigger evaluation on a completed matrix run.

**Test first:**
- `app/components/__tests__/EvaluateButton.test.tsx`
  - Renders when run has `completed` status
  - Disabled during evaluation
  - Shows progress during evaluation
  - Triggers results refresh after evaluation completes

---

## Phase 5: Results Dashboard

### Task 5.1 ⬜ — Results loader (TypeScript)

**What:** Parse `manifest.json` + `evaluation.json` into structured result objects.

**Test first:**
- `app/lib/__tests__/results-loader.test.ts`
  - Loads manifest and evaluation.json from run directory
  - Merges permutation params with evaluation results
  - Returns array of `{ params, loraFile, imageFile, status, inferenceTimeMs }`
  - Handles missing evaluation.json (returns results without images)

### Task 5.2 ⬜ — Results API endpoint

**What:** `app/api/results/route.ts` — browse results, fetch images.

**Test first:**
- `app/api/results/__tests__/route.test.ts`
  - GET returns list of all completed runs
  - GET with `?runId=X` returns detailed results for that run
  - Supports `?sort=param_name` query parameter
  - Supports `?filter=param_name:value` query parameter

### Task 5.3 ⬜ — Results grid component

**What:** `app/components/ResultsGrid.tsx` — grid of evaluation image cards.

**Test first:**
- `app/components/__tests__/ResultsGrid.test.tsx`
  - Renders one card per permutation result
  - Each card shows parameter values and evaluation image
  - Missing images show placeholder
  - Clicking card selects it for comparison

### Task 5.4 ⬜ — Filter/sort controls

**What:** UI controls for filtering and sorting results by parameter values.

**Test first:**
- `app/components/__tests__/ResultsFilters.test.tsx`
  - Shows filter dropdowns for each parameter dimension
  - Filtering narrows visible results
  - Sorting reorders results by selected parameter
  - Clearing filters shows all results

### Task 5.5 ⬜ — Side-by-side comparison view

**What:** Select 2+ results and display them side by side.

**Test first:**
- `app/components/__tests__/ComparisonView.test.tsx`
  - Renders selected results in a horizontal row
  - Each panel shows image, params, and LoRA file link
  - Minimum 2 selections required
  - Deselect removes from comparison

### Task 5.6 ⬜ — LoRA download links

**What:** Download links for each LoRA `.safetensors` file.

**Test first:**
- `app/components/__tests__/LoraDownload.test.tsx`
  - Renders download link pointing to correct file path
  - Link disabled when file does not exist

---

## Phase 6: Polish & Robustness

### Task 6.1 ⬜ — Training log streaming

**What:** Stream training output logs from child process to UI via Server-Sent Events or polling.

**Test first:**
- `app/lib/__tests__/log-streamer.test.ts`
  - Captures stdout/stderr from child process
  - Makes logs available line-by-line
  - Supports searching/filtering log lines

### Task 6.2 ⬜ — Log viewer UI

**What:** Scrollable, searchable log viewer panel.

**Test first:**
- `app/components/__tests__/LogViewer.test.tsx`
  - Renders log lines in order
  - Search input filters visible lines
  - Auto-scrolls to latest line
  - "Failed" lines highlighted in red

### Task 6.3 ⬜ — GPU VRAM monitoring

**What:** Periodic `nvidia-smi` query to report VRAM usage during training.

**Test first:**
- `app/lib/__tests__/vram-monitor.test.ts`
  - Parses nvidia-smi JSON output for VRAM used/total
  - Returns percentage used
  - Handles nvidia-smi failure gracefully

### Task 6.4 ⬜ — Pause/resume/cancel

**What:** Control signals for running matrix jobs.

**Test first:**
- `app/lib/__tests__/job-controller.test.ts`
  - `pause(jobId)` writes pause signal file
  - `resume(jobId)` removes pause signal file
  - `cancel(jobId)` writes cancel signal file and terminates process
  - Matrix trainer respects pause signal (waits between permutations)
  - Matrix trainer respects cancel signal (exits loop)

### Task 6.5 ⬜ — Parameter preset save/load

**What:** Save and load parameter configurations as JSON presets.

**Test first:**
- `app/lib/__tests__/preset-store.test.ts`
  - `savePreset(name, params)` stores preset
  - `loadPreset(name)` returns stored params
  - `listPresets()` returns all preset names
  - Overwriting preset name replaces old data
  - Deleting preset removes it

### Task 6.6 ⬜ — Error boundary and user-friendly errors

**What:** Global error boundaries and formatted error messages.

**Test first:**
- `app/components/__tests__/ErrorBoundary.test.tsx`
  - Catches render errors and shows fallback UI
  - Shows error message from API responses
  - Provides "retry" action where applicable

---

## Cross-Cutting Concerns

### Testing infrastructure setup ✅

- [x] **TypeScript tests:** Vitest configured with jsdom environment
- [x] **Python tests:** `pytest` added to `pyproject.toml` dev dependencies
- [x] **Test directories:** `tests/` (Python) and `__tests__/` (TypeScript, colocated)
- [x] **CI check:** `bunx vitest run` and `uv run pytest tests/` both pass

### Shared utilities to test early

| Utility | Test file | Key assertions |
|---|---|---|
| Path helpers (resolve model paths, output dirs) | `app/lib/__tests__/paths.test.ts` + `tests/test_paths.py` | Correct resolution on Windows and Linux |
| Child process runner (launch `uv run` commands) | `app/lib/__tests__/process-runner.test.ts` | Captures stdout/stderr, returns exit code, supports timeout, supports abort |
| File watcher (detect output file changes) | `app/lib/__tests__/file-watcher.test.ts` | Emits event on file creation, handles rapid changes |
