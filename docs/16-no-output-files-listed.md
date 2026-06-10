# Issue #16 — Completed Jobs Don't List Produced `.safetensors` Files 🔵 LOW

## What's Broken

When a training job completes, the job card in the Jobs panel shows the status as "completed" but doesn't list which `.safetensors` files were produced. The user has to know the output directory path and browse it manually to find the trained LoRA files.

## Where

**File:** `app/components/JobList.tsx`

The job card shows:
- LoRA name / job ID
- Status badge
- Network dim, alpha, epochs, resolution
- Start time
- Progress bar (for running jobs)
- Error message (for failed jobs)

It does NOT show:
- Output file paths
- List of produced `.safetensors` files
- Link to download the LoRA

**File:** `app/api/progress/[jobId]/route.ts`

The progress manifest includes `status`, `current_epoch`, `total_epochs`, `current_step`, `total_steps`, `avg_loss`, `error`, `exit_code` — but NOT a list of output files.

## What Needs to Happen

### Option A: Scan output dir for .safetensors files on job completion

In the progress API, when status is `completed`, also scan for output files:

```typescript
// In progress route, after reading manifest:
if (manifest.status === 'completed') {
  const outputFiles = fs.readdirSync(outputDir)
    .filter(f => f.endsWith('.safetensors'))
    .map(f => path.join(outputDir, f));
  
  return NextResponse.json({
    ...progressData,
    outputFiles,
  });
}
```

### Option B: Store output files in the job manifest

In `train_single.py`, after training completes, list output files and write to manifest:

```python
# In train_single.py, after training completes:
output_files = [f.name for f in output_dir.glob('*.safetensors')]
manifest['output_files'] = output_files
```

### Option C: Add output files to JobList UI

```typescript
{job.status === 'completed' && job.params.outputDir && (
  <div className="mt-2">
    <p className="text-xs text-slate-500 dark:text-slate-400">Output files:</p>
    <div className="flex flex-wrap gap-1 mt-1">
      {outputFiles.map(file => (
        <span key={file} className="text-xs font-mono bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 px-2 py-0.5 rounded">
          {file}
        </span>
      ))}
    </div>
  </div>
)}
```

## Testing

- Complete a training job
- Verify the job card lists the produced `.safetensors` files
- Verify clicking a file name triggers download (requires Issue #5 fix)
