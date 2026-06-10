# Issue #3 — Model Downloader Spawns Raw `python` Instead of `uv run python` 🔴 CRITICAL

## What's Broken

The model downloader spawns a raw `python` command to execute the inline download script. This project has **no standard `python` binary** — it uses `uv` for all Python execution. The download will fail with `python: command not found` (or whatever error the OS gives for a missing command).

## Where

**File:** `app/lib/model-downloader.ts`, line ~138

```typescript
const proc = spawn('python', [scriptFile, url, dest], {
  shell: false,
  env: spawnEnv,
});
```

## Root Cause

The project uses `uv` as the Python runtime manager (see pyproject.toml, uv.lock). All other Python scripts in the codebase correctly use `uv run python ...`:

- `app/api/train/route.ts` → `spawn('uv', ['run', 'python', ...])`
- `app/api/setup/route.ts` → `spawn('uv', ['run', 'python', ...])`
- `app/api/evaluate/route.ts` → `spawn('uv', ['run', 'python', ...])`

But `model-downloader.ts` uses bare `python`.

## What Needs to Happen

Change the spawn call to use `uv run python`:

```typescript
const proc = spawn('uv', ['run', 'python', scriptFile, url, dest], {
  shell: false,
  env: spawnEnv,
});
```

## Testing

1. Delete all models from `models/` directory
2. Navigate to Models tab
3. Click Download on any model
4. Verify download starts and progresses (no "command not found" error)
