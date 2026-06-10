# Issue #7 — Resolution Silently Clamped to 768–1024 with No User Warning 🟠 MEDIUM

## What's Broken

The `dataset_toml.py` script silently clamps the resolution to the range 768–1024. If a user enters 2048 or 512, the training will use a different resolution than they specified, with no warning.

## Where

**File:** `scripts/dataset_toml.py`, lines 45-50

```python
# Clamp resolution to Anima-supported bucket range
min_reso = 768
max_reso = 1024
bucket_steps = 16

clamped = max(min_reso, min(resolution, max_reso))
```

The UI (`AnimaTab.tsx`) has `min=256` on the resolution input — so a user can enter any value ≥ 256. Values below 768 or above 1024 are silently adjusted.

## Impact

- User enters 2048 expecting 2K training → gets 1024px training
- User enters 512 expecting faster training → gets 768px training (slower than expected)
- No log message indicates the clamping happened

## What Needs to Happen

### Option A: Warn in the Python script (minimum)

Add a log message when clamping occurs:

```python
if resolution != clamped:
    logger.warning(
        f"Resolution {resolution} clamped to {clamped} "
        f"(Anima supports {min_reso}-{max_reso}px)"
    )
```

### Option B: Validate in the UI (recommended)

Add validation in `AnimaTab.tsx`:

```typescript
// In validate()
if (params.resolution < 768 || params.resolution > 1024) {
  newErrors.resolution = `Resolution must be between 768 and 1024 (Anima limit)`;
}

// In the resolution input
{renderNumberInput('Resolution', 'resolution', 768, 16, 1024)}
```

Also update the `trainingSchema`:

```typescript
resolution: z.number().int().min(768).max(1024).default(1024),
```

### Option C: Both (best)

Validate in UI AND warn in Python as a safety net.

## Testing

- Enter 2048 → see validation error in UI
- Enter 512 → see validation error in UI
- Enter 1024 → accepted, no error
- Enter 768 → accepted, no error
