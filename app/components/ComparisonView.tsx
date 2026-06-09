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
      <div className="comparison-view">
        <p className="comparison-hint">
          Select at least 2 results to compare side by side
        </p>
      </div>
    );
  }

  return (
    <div className="comparison-view">
      <div className="comparison-row">
        {selectedResults.map(({ result, index, position }) => (
          <div key={index} className="comparison-panel" role="region">
            {/* Remove button */}
            {onDeselect && (
              <button
                className="remove-btn"
                onClick={() => onDeselect(index)}
              >
                Remove
              </button>
            )}

            {/* Evaluation image */}
            <div className="comparison-image">
              {result.imageFile ? (
                <img
                  src={`/output/${result.imageFile}`}
                  alt={`Result ${position + 1}`}
                />
              ) : (
                <div className="image-placeholder">No image</div>
              )}
            </div>

            {/* Parameter values */}
            <div className="comparison-params">
              {Object.entries(result.params).map(([key, value]) => (
                <div key={key} className="param-row">
                  <span className="param-name">{key}:</span>
                  <span className="param-value">
                    {typeof value === 'number' && value < 0.01
                      ? value.toExponential(1)
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>

            {/* LoRA file link */}
            {result.loraFile && (
              <div className="comparison-lora">
                <span className="lora-label">LoRA:</span>
                <span className="lora-file">{result.loraFile}</span>
              </div>
            )}

            {/* Inference time */}
            {result.inferenceTimeMs != null && (
              <div className="comparison-time">
                {result.inferenceTimeMs}ms
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
