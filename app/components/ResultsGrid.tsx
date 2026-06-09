'use client';

import { ResultEntry } from '../lib/results-loader';

interface ResultsGridProps {
  results: ResultEntry[];
  selectedIds?: number[];
  onSelectChange?: (ids: number[]) => void;
}

/**
 * Grid of evaluation image cards showing parameter values and results.
 */
export function ResultsGrid({
  results,
  selectedIds = [],
  onSelectChange,
}: ResultsGridProps) {
  const handleCardClick = (index: number) => {
    if (!onSelectChange) return;

    const newIds = selectedIds.includes(index)
      ? selectedIds.filter((id) => id !== index)
      : [...selectedIds, index];

    onSelectChange(newIds);
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {results.map((result, index) => {
        const isSelected = selectedIds.includes(index);

        return (
          <article
            key={index}
            role="article"
            onClick={() => handleCardClick(index)}
            className={`border rounded-lg overflow-hidden cursor-pointer transition-colors ${
              isSelected
                ? 'border-slate-900 dark:border-slate-100 ring-2 ring-slate-400 dark:ring-slate-500'
                : 'border-slate-200 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-500'
            } ${result.status === 'failed' ? 'bg-red-50 dark:bg-red-900/10' : 'bg-white dark:bg-slate-800'}`}
          >
            {/* Parameter values */}
            <div className="p-3 flex flex-wrap gap-1">
              {Object.entries(result.params).map(([key, value]) => (
                <span
                  key={key}
                  className="inline-block px-2 py-0.5 text-xs rounded-full bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300"
                >
                  {key}: {typeof value === 'number' ? formatNumber(value) : value}
                </span>
              ))}
            </div>

            {/* Evaluation image or placeholder */}
            <div className="px-3 pb-2">
              {result.imageFile ? (
                <img
                  src={`/output/${result.imageFile}`}
                  alt={`Evaluation result for ${result.loraFile}`}
                  role="img"
                  className="w-full rounded-md"
                />
              ) : (
                <div className="w-full h-40 flex items-center justify-center rounded-md bg-slate-100 dark:bg-slate-900 text-slate-400 dark:text-slate-500 text-sm">
                  No image
                </div>
              )}
            </div>

            {/* Status and timing */}
            <div className="px-3 pb-3 flex items-center justify-between">
              <span
                className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                  result.status === 'completed'
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                    : result.status === 'failed'
                    ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                    : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300'
                }`}
              >
                {result.status}
              </span>
              {result.inferenceTimeMs != null && (
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {result.inferenceTimeMs}ms
                </span>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
}

/**
 * Format a number: use compact scientific notation for very small values.
 */
function formatNumber(value: number): string {
  if (value !== 0 && Math.abs(value) < 0.01) {
    return value.toExponential(1);
  }
  return String(value);
}
