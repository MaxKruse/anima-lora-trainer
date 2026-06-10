# Issue #10 — Server Restart Loses Active Job Tracking 🟠 MEDIUM

## What's Broken

The `activeJobs` Map in `app/api/train/route.ts` is **in-memory only**. If the Next.js dev server restarts (hot reload, crash, manual restart), the map is cleared. Any running training jobs become "orphaned" — the child Python process keeps running, but:

1. The UI shows the job as `running` (from the file-based job store) but can't update progress
2. Cancel button doesn't work (cancel signal file is written, but the `proc.kill()` path is dead — see Issue #4)
3. The job never transitions to `completed` or `failed` in the job store
4. The Python process continues until training finishes or is manually killed

## Where

**File:** `app/api/train/route.ts`

```typescript
// In-memory state — lost on restart
const activeJobs = new Map<string, { params: TrainingParams; proc: any }>();
```

The file-based `JobStore` (in `jobs/`) persists job records, but it only stores static data (params, status). It doesn't track PIDs or process handles.

## What Needs to Happen

### Option A: Persist PID to disk (recommended)

Store the child process PID in the job file:

```typescript
// When creating a job:
const proc = spawn(cmd, args, { ... });
store.updateJobMeta(jobId, { pid: proc.pid });

// On server restart, scan for stale PIDs:
function recoverJobs() {
  const jobs = store.listJobs();
  for (const job of jobs) {
    if (job.status === 'running') {
      const pid = job.pid;
      if (pid && !isProcessAlive(pid)) {
        // Process died without completing
        store.updateStatus(jobId, 'failed');
        store.updateError(jobId, 'Process died (server restart or crash)');
      } else if (pid && isProcessAlive(pid)) {
        // Process still running — adopt it
        activeJobs.set(jobId, { params: job.params, pid });
      }
    }
  }
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0); // Sends no signal, just checks existence
    return true;
  } catch {
    return false;
  }
}
```

### Option B: Resume from manifest on restart

The Python training script writes `job_manifest.json` to the output directory. On server restart, scan all `running` jobs and check their manifests:

```typescript
// On startup:
function syncJobStatuses() {
  const jobs = store.listJobs();
  for (const job of jobs.filter(j => j.status === 'running')) {
    const manifest = readProgressManifest(job.params.outputDir);
    if (manifest?.status === 'completed') {
      store.updateStatus(job.id, 'completed');
    } else if (manifest?.status === 'failed') {
      store.updateStatus(job.id, 'failed');
      store.updateError(job.id, manifest.error);
    }
    // If still 'running' in manifest, the process is likely still alive
  }
}
```

### Option C: Both (best)

Combine PID tracking with manifest sync for robust recovery.

## Testing

1. Start a training job
2. Restart the Next.js dev server (Ctrl+C, then `bun dev`)
3. Navigate to Jobs → verify the job's status is correctly recovered (not stuck as "running" forever)
4. Verify cancel still works for recovered jobs
