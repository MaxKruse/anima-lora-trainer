'use client';

import { useState, useEffect, useCallback } from 'react';
import { ResultsGrid } from './ResultsGrid';
import { ResultsFilters } from './ResultsFilters';
import { ComparisonView } from './ComparisonView';
import { EvaluateButton } from './EvaluateButton';
import { LoraDownload } from './LoraDownload';
import { type ResultEntry } from '../lib/results-loader';

interface RunInfo {
  runId: string;
  total: number;
  completed: number;
  failed: number;
}

/**
 * Dashboard section for viewing training results and evaluations.
 * Lists completed runs, allows evaluation, and shows results grid with filters.
 */
export function ResultsDashboard() {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [results, setResults] = useState<ResultEntry[]>([]);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [sortBy, setSortBy] = useState('');
  const [filterBy, setFilterBy] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);

  // Load runs on mount
  const fetchRuns = useCallback(async () => {
    try {
      const res = await fetch('/api/results');
      if (res.ok) {
        const data = await res.json();
        setRuns(data.runs || []);
      }
    } catch {
      // Ignore fetch errors
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  // Load results when run selected
  const fetchResults = useCallback(async () => {
    if (!selectedRun) return;
    try {
      let url = `/api/results?runId=${encodeURIComponent(selectedRun)}`;
      if (sortBy) url += `&sort=${encodeURIComponent(sortBy)}`;
      if (Object.keys(filterBy).length > 0) {
        for (const [k, v] of Object.entries(filterBy)) {
          url += `&filter=${encodeURIComponent(k)}:${encodeURIComponent(v)}`;
        }
      }
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setResults(data.results || []);
      }
    } catch {
      // Ignore fetch errors
    }
  }, [selectedRun, sortBy, filterBy]);

  useEffect(() => {
    fetchResults();
    setSelectedIndices([]);
  }, [fetchResults]);

  function handleResultsRefresh() {
    fetchResults();
  }

  function handleSortChange(paramName: string) {
    setSortBy(paramName);
  }

  function handleFilterChange(filters: Record<string, string>) {
    setFilterBy(filters);
  }

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4">Training Results</h2>
        <p className="text-slate-500 dark:text-slate-400">Loading results...</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-6">Training Results</h2>

      {/* Run selector */}
      <section className="mb-8">
        <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
          Completed Runs
        </h3>

        {runs.length === 0 && (
          <p className="text-slate-500 dark:text-slate-400">
            No completed training runs found. Complete a training job to see results here.
          </p>
        )}

        <div className="space-y-2">
          {runs.map((run) => (
            <div
              key={run.runId}
              className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                selectedRun === run.runId
                  ? 'border-slate-900 dark:border-slate-100 bg-slate-50 dark:bg-slate-800'
                  : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-500'
              }`}
              onClick={() => {
                setSelectedRun(run.runId);
              }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-medium text-slate-900 dark:text-slate-100">
                    {run.runId}
                  </span>
                  <span className="text-sm text-slate-500 dark:text-slate-400 ml-3">
                    {run.completed}/{run.total} completed
                    {run.failed > 0 && ` · ${run.failed} failed`}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <EvaluateButton
                    runId={run.runId}
                    runStatus={run.completed === run.total ? 'completed' : 'running'}
                    onResultsRefresh={handleResultsRefresh}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Results view */}
      {selectedRun && (
        <section>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Results for {selectedRun}
            </h3>
            <button
              onClick={() => {
                setSelectedRun(null);
                setSortBy('');
                setFilterBy({});
              }}
              className="text-sm text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
            >
              Clear selection
            </button>
          </div>

          {results.length === 0 && (
            <p className="text-slate-500 dark:text-slate-400 mb-4">
              No results available. Run an evaluation to generate results.
            </p>
          )}

          {/* Filters */}
          {results.length > 0 && (
            <div className="mb-4">
              <ResultsFilters
                results={results}
                onSortChange={handleSortChange}
                onFilterChange={handleFilterChange}
              />
            </div>
          )}

          {/* Results grid */}
          {results.length > 0 && (
            <ResultsGrid
              results={results}
              selectedIds={selectedIndices}
              onSelectChange={setSelectedIndices}
            />
          )}

          {/* Download links for each result */}
          {results.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2">
                Download LoRA files
              </h4>
              <div className="flex flex-wrap gap-2">
                {results
                  .filter(r => r.loraFile)
                  .map((result, idx) => (
                    <LoraDownload
                      key={idx}
                      runId={selectedRun}
                      loraFile={result.loraFile!}
                      exists={result.status === 'completed'}
                    />
                  ))}
              </div>
            </div>
          )}

          {/* Comparison view */}
          {selectedIndices.length >= 2 && results.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">
                Side-by-Side Comparison
              </h3>
              <ComparisonView
                results={results}
                selectedIndices={selectedIndices}
                onDeselect={(idx) =>
                  setSelectedIndices((prev) => prev.filter((i) => i !== idx))
                }
              />
            </div>
          )}
        </section>
      )}
    </div>
  );
}
