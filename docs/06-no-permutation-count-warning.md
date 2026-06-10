# Issue #6 — No Permutation Count Warning in Matrix Mode 🟡 HIGH

## What's Broken

In Matrix mode, the user selects multiple values for each parameter (e.g., dims: 8, 16, 32, 64, 128, 256 = 6 values × alphas: 4, 8, 16, 32, 64 = 5 values × learning rates: 4 values = **120 permutations**). There's no warning about how many training jobs this will create.

A user could easily select 6 × 5 × 4 × 4 × 5 × 2 × 2 = **9,600 permutations** without realizing it. Each permutation is a full training run that could take 10-30 minutes.

## Where

**File:** `app/components/AnimaTab.tsx`

The `matrixValues` state tracks selected values per parameter, but no calculation or display of the total permutation count exists.

**File:** `app/components/MatrixToggle.tsx`

The `MatrixToggle` component accepts a `permutationCount` prop and displays it — but nothing ever passes a real count:

```typescript
// MatrixToggle.tsx — this prop is never populated
{mode === 'matrix' && permutationCount !== undefined && permutationCount > 0 && (
  <span className="text-sm text-slate-500 dark:text-slate-400 ml-2">
    ({permutationCount} permutations)
  </span>
)}
```

## What Needs to Happen

### 1. Calculate permutation count in AnimaTab

```typescript
// In AnimaTab.tsx
const permutationCount = useMemo(() => {
  if (!matrixMode) return 0;
  return Object.values(matrixValues).reduce((total, values) => total * values.length, 1);
}, [matrixMode, matrixValues]);
```

### 2. Display the count prominently

Show it near the "Start Training" button:

```typescript
{matrixMode && (
  <div className={`p-3 rounded-lg mb-4 ${
    permutationCount > 100
      ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
      : permutationCount > 20
        ? 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800'
        : 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800'
  }`}>
    <p className="text-sm font-medium">
      {permutationCount} training permutations
    </p>
    {permutationCount > 100 && (
      <p className="text-xs text-red-600 dark:text-red-400 mt-1">
        Warning: This will create {permutationCount} separate training jobs.
        Estimated time: {Math.round(permutationCount * 15 / 60)} hours.
      </p>
    )}
  </div>
)}
```

### 3. Pass count to MatrixToggle

```typescript
<MatrixToggle
  mode={matrixMode}
  onChange={handleMatrixModeChange}
  permutationCount={permutationCount}
/>
```

### 4. Optional: Block submit above threshold

```typescript
const MAX_PERMS = 500;
if (matrixMode && permutationCount > MAX_PERMS) {
  setErrors(prev => ({ ...prev, _submit: `Too many permutations (${permutationCount}). Maximum is ${MAX_PERMS}.` }));
  return;
}
```

## Testing

- Select 3 values for dim, 2 for alpha, 2 for lr → verify "12 permutations" displayed
- Select many values → verify warning appears
- Verify MatrixToggle shows the count
- Verify submit is blocked (if threshold implemented)
