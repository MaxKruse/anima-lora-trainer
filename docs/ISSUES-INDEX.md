# Issues Index — LoRA Matrix Trainer

Quick reference to all known issues, sorted by severity.

| # | Severity | File | Summary |
|---|----------|------|---------|
| [1](./01-matrix-training-dead-code.md) | 🔴 CRITICAL | Matrix training UI never calls `/api/train/matrix` |
| [2](./02-results-ui-unreachable.md) | 🔴 CRITICAL | Results/Evaluation components are orphaned — no UI entry point |
| [3](./03-model-downloader-wrong-python.md) | 🔴 CRITICAL | Model downloader spawns raw `python` instead of `uv run python` |
| [4](./04-active-jobs-proc-null.md) | 🟡 HIGH | `activeJobs` proc reference is always `null` — direct kill doesn't work |
| [5](./05-lora-download-missing-api.md) | 🟡 HIGH | `LoraDownload` links to non-existent `/api/download` endpoint |
| [6](./06-no-permutation-count-warning.md) | 🟡 HIGH | Matrix mode can silently produce hundreds of jobs |
| [7](./07-resolution-silent-clamp.md) | 🟠 MEDIUM | Resolution silently clamped to 768–1024 with no user warning |
| [8](./08-no-training-confirmation.md) | 🟠 MEDIUM | No confirmation or duration estimate before starting training |
| [9](./09-single-job-limit.md) | 🟠 MEDIUM | Only one job allowed at a time — no queue |
| [10](./10-server-restart-orphaned-jobs.md) | 🟠 MEDIUM | Server restart loses active job tracking |
| [11](./11-cancel-delay.md) | 🔵 LOW | Cancel is delayed (signal file polling) |
| [12](./12-abort-button-hidden.md) | 🔵 LOW | Download abort button only visible on hover |
| [13](./13-traintabs-overflow.md) | 🔵 LOW | `px-16 py-8` padding causes horizontal scroll on small screens |
| [14](./14-datasets-browser-limited.md) | 🔵 LOW | Datasets browser limited to project `datasets/` folder only |
| [15](./15-non-anima-models-empty.md) | 🔵 LOW | Non-Anima model manifests are empty — Models tab shows nothing |
| [16](./16-no-output-files-listed.md) | 🔵 LOW | Completed jobs don't list produced `.safetensors` files |
