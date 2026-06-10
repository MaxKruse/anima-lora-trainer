# Issue #12 — Download Abort Button Only Visible on Hover 🔵 LOW

## What's Broken

The abort button on the circular progress indicator during model downloads is hidden by default and only appears on hover (`group-hover:opacity-100`). This makes it:

- Invisible on touch devices (no hover)
- Easy to miss on desktop
- Poor accessibility (no keyboard focus reveals it)

## Where

**File:** `app/components/ModelDownloader.tsx`

```typescript
// CircularProgress component
{canAbort && (
  <button
    onClick={onAbort}
    className="absolute inset-0 w-full h-full flex items-center justify-center bg-red-500 rounded opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
    title="Abort download"
    aria-label="Abort download"
  >
    <svg ...>
```

The parent div has `className="group"` which enables `group-hover`:

```typescript
<div className="group p-4 border ...">
```

## What Needs to Happen

### Option A: Always show abort button when downloading (simplest)

```typescript
{canAbort && (
  <button
    onClick={onAbort}
    className="absolute top-1 right-1 w-6 h-6 flex items-center justify-center bg-red-500 rounded text-white text-xs hover:bg-red-600 transition-colors cursor-pointer z-10"
    title="Abort download"
    aria-label="Abort download"
  >
    ✕
  </button>
)}
```

### Option B: Show a separate abort button below the progress

```typescript
{model.status === 'downloading' && model.canAbort && (
  <button onClick={() => handleAbort(model.name)} className="text-xs text-red-600 hover:text-red-800 mt-1">
    Abort download
  </button>
)}
```

## Testing

- Start a model download
- Verify abort button is visible without hovering
- Verify it works on touch devices
