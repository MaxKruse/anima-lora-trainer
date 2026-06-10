# Issue #2 — Results/Evaluation UI is Completely Unreachable 🔴 CRITICAL

## What's Broken

Six components and two API endpoints exist for viewing and comparing training results. **None of them are rendered anywhere in the application.** There is no "Results" tab, no results section in the Jobs panel, and no way for a user to navigate to evaluation or comparison views.

## Orphaned Components

| Component | Purpose | Rendered? |
|-----------|---------|-----------|
| `ResultsGrid.tsx` | Grid of evaluation image cards | ❌ No |
| `ComparisonView.tsx` | Side-by-side comparison of 2+ results | ❌ No |
| `ResultsFilters.tsx` | Filter/sort controls for results | ❌ No |
| `EvaluateButton.tsx` | Button to trigger evaluation on completed runs | ❌ No |
| `LoraDownload.tsx` | Download link for `.safetensors` files | ❌ No |
| `LogViewer.tsx` | Searchable log viewer | ✅ Yes (in JobList) |

## Orphaned API Endpoints

| Endpoint | Purpose | Called from UI? |
|----------|---------|-----------------|
| `GET /api/results` | List runs or fetch detailed results | ❌ No |
| `GET /api/results?runId=X` | Detailed results for a run | ❌ No |
| `POST /api/evaluate` | Start evaluation for a run | ❌ No |
| `GET /api/evaluate?runId=X` | Fetch evaluation results | ❌ No |

## Where It Should Live

Per the original plan (PLAN.md §7.4), the results viewer should be accessible after training completes. Two options:

### Option A: New "Results" Section (recommended)

Add a 5th section to the Dashboard nav: `setup | models | train | jobs | results`

The Results section would:
1. List all completed runs (from `GET /api/results`)
2. Show `EvaluateButton` for each completed run
3. After evaluation, show `ResultsFilters` + `ResultsGrid`
4. Support multi-select → `ComparisonView`
5. Show `LoraDownload` links on each card

### Option B: Integrate into Jobs Panel

Add an "Evaluate" button and results section to each completed job card in `JobList.tsx`. Simpler but mixes concerns.

## What Needs to Happen

### Step 1: Add Results section to Dashboard

```typescript
// Dashboard.tsx
type DashboardSection = 'setup' | 'models' | 'train' | 'jobs' | 'results';

const sections = [
  // ... existing ...
  { key: 'results', label: 'Results' },
];
```

### Step 2: Create a ResultsDashboard component

```typescript
// app/components/ResultsDashboard.tsx
export function ResultsDashboard() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [results, setResults] = useState<ResultEntry[]>([]);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [sortBy, setSortBy] = useState('');
  const [filterBy, setFilterBy] = useState<Record<string, string>>({});

  // Load runs on mount
  useEffect(() => {
    fetch('/api/results').then(r => r.json()).then(d => setRuns(d.runs || []));
  }, []);

  // Load results when run selected
  useEffect(() => {
    if (!selectedRun) return;
    let url = `/api/results?runId=${selectedRun}`;
    if (sortBy) url += `&sort=${sortBy}`;
    if (Object.keys(filterBy).length > 0) {
      for (const [k, v] of Object.entries(filterBy)) {
        url += `&filter=${k}:${v}`;
      }
    }
    fetch(url).then(r => r.json()).then(d => setResults(d.results || []));
  }, [selectedRun, sortBy, filterBy]);

  return (
    <div>
      <h2>Training Results</h2>

      {/* Run selector */}
      <div className="runs-list">
        {runs.map(run => (
          <div key={run.runId} onClick={() => setSelectedRun(run.runId)}>
            {run.runId} — {run.completed}/{run.total} completed
            <EvaluateButton
              runId={run.runId}
              runStatus="completed"
              onResultsRefresh={() => /* reload */}
            />
          </div>
        ))}
      </div>

      {/* Results view */}
      {selectedRun && (
        <>
          <ResultsFilters
            results={results}
            onSortChange={setSortBy}
            onFilterChange={setFilterBy}
          />
          <ResultsGrid
            results={results}
            selectedIds={selectedIndices}
            onSelectChange={setSelectedIndices}
          />
          {selectedIndices.length >= 2 && (
            <ComparisonView
              results={results}
              selectedIndices={selectedIndices}
              onDeselect={(idx) => setSelectedIndices(prev => prev.filter(i => i !== idx))}
            />
          )}
        </>
      )}
    </div>
  );
}
```

### Step 3: Wire into Dashboard

```typescript
{activeSection === 'results' && <ResultsDashboard />}
```

### Step 4: Fix LoraDownload (see Issue #5)

The `LoraDownload` component references `/api/download` which doesn't exist. Either create the endpoint or serve files directly from the `output/` directory.

## Testing

- Complete a training job → navigate to Results → see the run listed
- Click Evaluate → see evaluation progress → see results grid populate
- Filter/sort results → verify grid updates
- Select 2+ cards → see comparison view
- Click download link → verify file downloads
