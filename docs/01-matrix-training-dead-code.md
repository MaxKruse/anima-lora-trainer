# Issue #1 — Matrix Training UI Never Calls `/api/train/matrix` 🔴 CRITICAL

## What's Broken

The Matrix mode toggle exists in the UI and renders `MultiSelectDropdown` components for multi-value parameters. However, when the user submits the form, **it always calls `/api/train` (single training)** — never `/api/train/matrix`.

The matrix training endpoint and Python script (`scripts/matrix_trainer.py`) are fully implemented and tested. But the UI code path that would reach them doesn't exist.

## Where

| File | What |
|------|------|
| `app/components/Dashboard.tsx` | `handleTrainingSubmit()` always POSTs to `/api/train` |
| `app/components/AnimaTab.tsx` | `handleSubmit()` validates against `trainingSchema` (single params) and calls `onSubmit(result.data)` |
| `app/components/MatrixToggle.tsx` | Toggles `matrixMode` state and fires `onChange` |
| `app/api/train/route.ts` | Single training endpoint (always called) |
| `app/api/train/matrix/route.ts` | Matrix training endpoint (never called) |

## Root Cause

`Dashboard.tsx` line ~165 — the submit handler:

```typescript
const handleTrainingSubmit = useCallback(
  async (params: TrainingParams) => {
    // ...
    const res = await fetch('/api/train', {   // ← ALWAYS single
      method: 'POST',
      body: JSON.stringify(params),
    });
    // ...
  },
  [],
);
```

It receives `TrainingParams` (single-run schema) and posts to `/api/train`. There's no branching on `matrixMode`.

Meanwhile, `AnimaTab.tsx` collects matrix values in `matrixValues` state (a `Record<string, string[]>`), but `handleSubmit()` never uses it — it always validates and submits `params` (single values).

## What Needs to Happen

### Option A: Fix in AnimaTab (recommended)

1. **Pass `matrixMode` through to `handleSubmit`** — already available as a prop
2. **When `matrixMode === true`:**
   - Collect `matrixValues` (the multi-select state) into `paramRanges` format expected by `/api/train/matrix`
   - Collect non-matrix fields (trainingImages, loraName, mixedPrecision, timestepSampling, optimization checkboxes, captionTagDropoutRate) into `baseParams`
   - POST to `/api/train/matrix` instead of `/api/train`
3. **When `matrixMode === false`:** keep existing single training behavior

### Option B: Fix in Dashboard

Route the submit based on `matrixMode` at the Dashboard level. This requires `AnimaTab` to emit a different event type for matrix mode.

### Concrete Changes

**`app/components/AnimaTab.tsx`** — `handleSubmit()`:

```typescript
async function handleSubmit() {
  if (!validate()) return;
  if (nameAvailable === false) return;

  setSubmitting(true);
  try {
    if (matrixMode) {
      // Build matrix payload
      const paramRanges: Record<string, string> = {};
      for (const [key, values] of Object.entries(matrixValues)) {
        paramRanges[key] = values.join(',');
      }
      const baseParams = {
        trainingImages: isManagedExternally ? trainingImagesPath : params.trainingImages,
        loraName: params.loraName,
        mixedPrecision: params.mixedPrecision,
        timestepSampling: params.timestepSampling,
        gradientCheckpointing: params.gradientCheckpointing,
        cacheLatents: params.cacheLatents,
        cacheTextEncoder: params.cacheTextEncoder,
        captionTagDropoutRate: params.captionTagDropoutRate,
      };

      const res = await fetch('/api/train/matrix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paramRanges, baseParams }),
      });

      const data = await res.json();
      if (!res.ok) {
        setErrors({ _submit: data.error || 'Matrix training failed' });
      } else {
        // Navigate to jobs, show success
      }
    } else {
      // Existing single training path
      // ...
    }
  } finally {
    setSubmitting(false);
  }
}
```

**`app/components/MatrixToggle.tsx`** — add permutation count calculation:
- Listen to `matrixValues` changes and compute Cartesian product count
- Display warning if count > threshold (see Issue #6)

## Testing

- Toggle to Matrix mode, select multiple values for dim/alpha/lr
- Verify request goes to `/api/train/matrix` with correct `paramRanges` and `baseParams`
- Verify single mode still works unchanged
- Verify name availability check still works in matrix mode
