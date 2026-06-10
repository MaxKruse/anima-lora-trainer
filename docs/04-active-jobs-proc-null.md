# Issue #4 — `activeJobs` proc Reference is Always `null` 🟡 HIGH

## What's Broken

When a training job is cancelled, the code tries to kill the child process directly with `proc.kill('SIGTERM')`. But the `proc` stored in `activeJobs` is always `null`, so the kill silently fails. The cancel signal file mechanism works, but the direct process termination is dead code.

## Where

**File:** `app/api/train/route.ts`

```typescript
// Line ~145 — job is tracked with proc: null
activeJobs.set(jobId, { params, proc: null });

// Line ~165 — cancelJob tries to kill the proc
export function cancelJob(jobId: string): boolean {
  const activeJob = activeJobs.get(jobId);
  // ...
  if (activeJob.proc) {        // ← always falsy
    try {
      activeJob.proc.kill('SIGTERM');
    } catch { /* ignore */ }
  }
}
```

The `launchTraining()` function creates the child process but never updates `activeJobs` with the actual `proc` reference:

```typescript
function launchTraining(jobId: string, params: TrainingParams, outputDir: string): Promise<void> {
  return new Promise((resolve) => {
    // ...
    const proc = spawn(cmd, args, { shell: true, cwd: PROJECT_ROOT });
    // ← proc is created but never stored in activeJobs!
    // ...
  });
}
```

## Root Cause

`launchTraining()` is a separate function that creates the `spawn` process in its own scope. It never passes the `proc` reference back to update `activeJobs`.

## Impact

- Cancel still works (signal file is written, Python script polls for it)
- But there's a ~1 second delay before the Python script detects the cancel
- The `SIGTERM` kill would be instant if it worked
- If the Python script's cancel check is buggy, the process would never die

## What Needs to Happen

### Option A: Store proc in activeJobs (simplest)

```typescript
// In POST handler, before calling launchTraining:
activeJobs.set(jobId, { params, proc: null });

// In launchTraining, after spawn:
const proc = spawn(cmd, args, { ... });
activeJobs.set(jobId, { params, proc });  // ← store the actual proc
```

### Option B: Return proc from launchTraining

```typescript
function launchTraining(...): ChildProcess {
  // ...
  const proc = spawn(cmd, args, { ... });
  activeJobs.set(jobId, { params, proc });
  // ...
  return proc;
}
```

## Testing

1. Start a training job
2. Immediately cancel it
3. Verify the process terminates quickly (not waiting for next cancel-signal poll cycle)
