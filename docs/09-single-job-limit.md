# Issue #9 — Only One Job Allowed at a Time, No Queue 🟠 MEDIUM

## What's Broken

The `/api/train` endpoint rejects any new training request if a job is already running:

```typescript
// app/api/train/route.ts
if (activeJobs.size > 0) {
  const [currentJobId] = activeJobs.keys();
  return NextResponse.json(
    { error: 'A training job is already running', currentJobId },
    { status: 409 }
  );
}
```

There's no job queue. If a user wants to train 5 different LoRAs sequentially, they must manually start each one after the previous completes.

## Impact

- For single training: minor inconvenience (user can just wait and click again)
- For matrix training: not an issue (matrix trainer handles permutations internally)
- For power users who want to queue multiple different configs: blocked

## What Needs to Happen

### Option A: Simple queue (recommended for now)

Add a queue array alongside `activeJobs`:

```typescript
const jobQueue: TrainingParams[] = [];

// In POST handler:
if (activeJobs.size > 0) {
  jobQueue.push(params);
  return NextResponse.json({
    jobId: 'queued',
    status: 'queued',
    position: jobQueue.length,
    message: 'Job added to queue. It will start when the current job finishes.',
  });
}

// In launchTraining's proc.on('close'):
if (jobQueue.length > 0) {
  const nextParams = jobQueue.shift()!;
  launchNextQueuedJob(nextParams);
}
```

### Option B: Allow concurrent jobs

Remove the `activeJobs.size > 0` check and allow multiple concurrent training processes. This requires:
- Updating `activeJobs` to properly track all procs (see Issue #4)
- Ensuring VRAM can handle multiple training processes
- Adding per-job resource limits

### Option C: Just improve the error message

Minimum fix — tell the user which job is running and that they can queue:

```typescript
return NextResponse.json({
  error: 'A training job is already running',
  currentJobId,
  hint: 'Wait for the current job to complete, or use Matrix mode to train multiple configurations at once.',
}, { status: 409 });
```

## Testing

- Start a job → try to start another → see queue message (Option A) or improved error (Option C)
- First job completes → second job starts automatically (Option A)
