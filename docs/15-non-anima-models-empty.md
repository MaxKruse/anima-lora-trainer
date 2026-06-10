# Issue #15 — Non-Anima Model Manifests Are Empty 🔵 LOW

## What's Broken

The model manifest defines empty arrays for all non-Anima model types (FLUX, SD3, SDXL, SD 1.5, Hunyuan, Lumina). When the Models tab loads, it resolves the Anima manifest (which works). But if a user ever switches to a non-Anima model type, the Models tab shows nothing because the manifest is empty.

This is expected for now (only Anima is implemented), but the Models tab doesn't indicate this — it just shows the Anima models regardless of which training tab is active.

## Where

**File:** `app/lib/model-manifest.ts`

```typescript
const MODEL_MANIFESTS: Record<ModelType, ModelEntry[]> = {
  anima: ANIMA_MODELS,   // 3 entries
  flux: [],              // empty
  sd3: [],               // empty
  sdxl: [],              // empty
  sd15: [],              // empty
  hunyuan: [],           // empty
  lumina: [],            // empty
};
```

**File:** `app/components/ModelDownloader.tsx`

Always fetches Anima models:

```typescript
const manifest = await getResolvedModelManifest('anima');  // hardcoded
```

## Impact

- No functional impact — only Anima training is implemented
- If/when other model types are added, the Models tab needs to be aware of the active model type

## What Needs to Happen

### Option A: Pass active model type to ModelDownloader

```typescript
// Dashboard.tsx
<ModelDownloader modelType={activeModelType} />

// ModelDownloader.tsx
const manifest = await getResolvedModelManifest(modelType);
```

### Option B: Show a message for empty manifests

```typescript
{models.length === 0 && (
  <p className="text-slate-500 dark:text-slate-400">
    No models available for {modelType}. Model downloads will be available when {modelType} training is implemented.
  </p>
)}
```

### Option C: Leave as-is

Only Anima is supported. The Models tab works for Anima. When other models are added, this will be fixed naturally.

## Recommendation

Option C for now. Document that the Models tab is Anima-only.
