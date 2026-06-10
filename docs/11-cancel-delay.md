# Issue #11 — Cancel is Delayed (Signal File Polling) 🔵 LOW

## What's Broken

When a user cancels a training job, the API writes a `.cancel` signal file and the Python training script polls for it. The polling interval is ~1 second (in `train_single.py`), so there's a delay between clicking "Cancel" and the training actually stopping.

## Where

**File:** `scripts/train_single.py`

```python
# Main loop — checks cancel signal every ~1 second
while proc.poll() is None:
    if job_id and check_cancel_signal(job_id):
        logger.info("Cancel signal detected — terminating training")
        proc.terminate()
        cancelled = True
        break
    time.sleep(1)  # ← 1 second polling interval
```

**File:** `app/api/jobs/[jobId]/cancel/route.ts`

```typescript
// Writes cancel file — Python will detect it on next poll
const cancelPath = path.join(PROJECT_ROOT, 'jobs', `${jobId}.cancel`);
fs.writeFileSync(cancelPath, new Date().toISOString());
```

## Impact

- Training continues for up to ~1 second after cancel
- During that second, GPU resources are still consumed
- Not a functional bug, just a UX imperfection

## What Needs to Happen

### Option A: Kill the process directly (fix Issue #4 first)

If Issue #4 is fixed (proc reference stored correctly), the cancel route can also send `SIGTERM`:

```typescript
// In cancel route, after writing signal file:
const activeJob = activeJobs.get(jobId);
if (activeJob?.proc) {
  activeJob.proc.kill('SIGTERM');  // Immediate termination
}
```

### Option B: Reduce polling interval

Change `time.sleep(1)` to `time.sleep(0.1)` (100ms). Minor improvement.

### Option C: Use a pipe/socket for instant signal

Overkill for this use case. Skip.

## Recommendation

Fix Issue #4 first. The direct `proc.kill()` makes cancel instant.
