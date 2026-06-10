# Issue #14 — Datasets Browser Limited to Project `datasets/` Folder Only 🔵 LOW

## What's Broken

The "Browse Datasets" button in the Train section only lists subdirectories inside the project's `datasets/` folder. Users who have training images elsewhere on their filesystem must manually type the full path into the directory picker.

## Where

**File:** `app/api/datasets/route.ts`

```typescript
const DATASETS_DIR = path.resolve(process.cwd(), 'datasets');

export async function GET() {
  // Only reads from DATASETS_DIR
  const entries = fs.readdirSync(DATASETS_DIR, { withFileTypes: true });
  // ...
}
```

**File:** `app/components/Dashboard.tsx`

The "Browse Datasets" button calls `GET /api/datasets` and populates a dropdown.

## Impact

- Minor inconvenience — user can still type the path manually
- The `datasets/` folder exists and has a `mari_setogaya` dataset, so it works for demo purposes
- Not a blocker

## What Needs to Happen

### Option A: Add common image directories to the browser

Include additional paths like Pictures, Desktop, or recently-used directories:

```typescript
const BROWSABLE_DIRS = [
  path.resolve(process.cwd(), 'datasets'),
  path.join(os.homedir(), 'Pictures'),
  path.join(os.homedir(), 'Desktop'),
];
```

### Option B: Add a "custom path" field

Allow the user to add custom browse paths to their config.

### Option C: Leave as-is

The manual path input works fine. The browser is a convenience feature, not a requirement.

## Recommendation

Option C for now. If users complain, implement Option A.
