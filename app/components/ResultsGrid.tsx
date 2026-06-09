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
    <div className="results-grid">
      {results.map((result, index) => {
        const isSelected = selectedIds.includes(index);

        return (
          <article
            key={index}
            role="article"
            className={`result-card ${isSelected ? 'selected' : ''} ${result.status === 'failed' ? 'failed' : ''}`}
            onClick={() => handleCardClick(index)}
          >
            {/* Parameter values */}
            <div className="result-params">
              {Object.entries(result.params).map(([key, value]) => (
                <span key={key} className="param-tag">
                  {key}: {typeof value === 'number' ? formatNumber(value) : value}
                </span>
              ))}
            </div>

            {/* Evaluation image or placeholder */}
            <div className="result-image">
              {result.imageFile ? (
                <img
                  src={`/output/${result.imageFile}`}
                  alt={`Evaluation result for ${result.loraFile}`}
                  role="img"
                />
              ) : (
                <div className="image-placeholder">No image</div>
              )}
            </div>

            {/* Status and timing */}
            <div className="result-meta">
              <span className={`status-${result.status}`}>{result.status}</span>
              {result.inferenceTimeMs != null && (
                <span className="inference-time">
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
