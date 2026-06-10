# Issue #8 — No Confirmation or Duration Estimate Before Starting Training 🟠 MEDIUM

## What's Broken

Clicking "Start Training" immediately launches the training job with no confirmation dialog, no warning about expected duration, and no summary of what's about to happen.

A training run can take anywhere from 5 minutes to many hours depending on parameters (epochs, batch size, dataset size, GPU). Users should be warned before committing.

## Where

**File:** `app/components/AnimaTab.tsx`

```typescript
<button type="submit" disabled={submitting}>
  {submitting ? 'Starting...' : 'Start Training'}
</button>
```

The form's `onSubmit` calls `handleSubmit()` which calls `onSubmit(result.data)` → Dashboard's `handleTrainingSubmit` → POST to `/api/train`. No intermediate confirmation.

## What Needs to Happen

### Add a confirmation modal/step

```typescript
// In AnimaTab.tsx, before submitting:
function handleSubmit() {
  if (!validate()) return;
  if (nameAvailable === false) return;

  // Show confirmation
  const estimatedTime = estimateTrainingTime(params);
  const message = `Start training "${params.loraName}"?

  Epochs: ${params.epochs}
  Batch size: ${params.batchSize}
  Estimated time: ~${estimatedTime}

  This will use your GPU exclusively.`;

  if (!confirm(message)) return;

  // ... proceed with submit
}

function estimateTrainingTime(params: typeof DEFAULT_PARAMS & { trainingImages: string; loraName: string }): string {
  // Rough estimate: ~15 min per epoch for typical dataset
  const minutesPerEpoch = 15;
  const totalMinutes = params.epochs * minutesPerEpoch;
  if (totalMinutes < 60) return `${totalMinutes} minutes`;
  return `${Math.round(totalMinutes / 60)} hours ${totalMinutes % 60} minutes`;
}
```

### Or: Two-step submit (better UX)

1. First click → show summary modal with params, estimated time, and "Confirm" / "Cancel" buttons
2. Second click (Confirm) → actually submit

## Testing

- Click "Start Training" → see confirmation with params summary
- Click Cancel → no job started
- Click Confirm → job starts normally
