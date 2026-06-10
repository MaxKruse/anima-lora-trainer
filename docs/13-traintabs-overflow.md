# Issue #13 — `px-16 py-8` Padding Causes Horizontal Scroll on Small Screens 🔵 LOW

## What's Broken

The `TrainTabs` component has `px-16` (4rem = 64px on each side) and `py-8` (2rem = 32px vertical) padding inside a `max-w-screen-xl` container. On screens ≤ 1280px, this leaves very little horizontal space for the 3-column grid of parameter inputs, causing horizontal scrolling.

## Where

**File:** `app/components/TrainTabs.tsx`

```typescript
return (
  <div className="max-w-screen-xl mx-auto px-16 py-8">
    {/* Model type tabs */}
    <div className="mb-6 border-b ...">
      {/* ... */}
    </div>

    {/* Tab content — AnimaTab has grid grid-cols-3 */}
    <div>
      {IMPLEMENTED_MODELS.includes(activeTab) ? (
        <AnimaTab ... />
      ) : (
        <ComingSoonTab ... />
      )}
    </div>
  </div>
);
```

## What Needs to Happen

Reduce padding and make it responsive:

```typescript
return (
  <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
```

Or move the padding to the Dashboard level (where it already exists as `max-w-7xl mx-auto px-4 sm:px-6 lg:px-8`) and remove it from TrainTabs entirely:

```typescript
return (
  <div className="max-w-screen-xl mx-auto">
```

The Dashboard already wraps content in `max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8`, so the TrainTabs padding is redundant and excessive.

## Testing

- Resize browser to 1366px width → verify no horizontal scroll
- Resize to 1024px → verify 3-column grid stacks or scrolls gracefully
