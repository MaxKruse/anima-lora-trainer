'use client';

import { ResultEntry } from '../lib/results-loader';

interface ComparisonViewProps {
  results: ResultEntry[];
  selectedIndices: number[];
  onDeselect?: (index: number) => void;
}

/**
 * Side-by-side comparison view for 2+ selected results.
 */
export function ComparisonView({
  results,
  selectedIndices,
  onDeselect,
}: ComparisonViewProps) {
  const selectedResults = selectedIndices
    .map((idx, position) => ({ result: results[idx], index: idx, position }))
    .filter(({ result }) => result != null);

  // Require minimum 2 selections
  if (selectedResults.length < 2) {
    return (
      <div className="p-6 text-center text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800">
        Select at least 2 results to compare side by side
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {selectedResults.map(({ result, index, position }) => (
          <div
            key={index}
            role="region"
            className="border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 overflow-hidden"
          >
            {/* Header with remove button */}
            <div className="flex items-center justify-between px-3 pt-3">
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Result {position + 1}
              </span>
              {onDeselect && (
                <button
                  onClick={() => onDeselect(index)}
                  className="text-xs text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                >
                  Remove
                </button>
              )}
            </div>

            {/* Evaluation image */}
            <div className="px-3 mt-2">
              {result.imageFile ? (
                <img
                  src={`/output/${result.imageFile}`}
                  alt={`Result ${position + 1}`}
                  className="w-full rounded-md"
                />
              ) : (
                <div className="w-full h-40 flex items-center justify-center rounded-md bg-slate-100 dark:bg-slate-900 text-slate-400 dark:text-slate-500 text-sm">
                  No image
                </div>
              )}
            </div>

            {/* Parameter values */}
            <div className="p-3 space-y-1">
              {Object.entries(result.params).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="text-slate-500 dark:text-slate-400">{key}:</span>
                  <span className="font-medium text-slate-900 dark:text-slate-100">
                    {typeof value === 'number' && value < 0.01
                      ? value.toExponential(1)
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>

            {/* LoRA file link */}
            {result.loraFile && (
              <div className="px-3 pb-2">
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  LoRA: <span className="font-mono text-slate-700 dark:text-slate-300">{result.loraFile}</span>
                </span>
              </div>
            )}

            {/* Inference time */}
            {result.inferenceTimeMs != null && (
              <div className="px-3 pb-3">
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {result.inferenceTimeMs}ms
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
