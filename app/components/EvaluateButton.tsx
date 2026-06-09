'use client';

import { useState, useCallback, useEffect } from 'react';

interface EvaluateButtonProps {
  runId: string;
  runStatus: string;
  onResultsRefresh?: () => void;
}

/**
 * Button to trigger evaluation on a completed matrix run.
 */
export function EvaluateButton({
  runId,
  runStatus,
  onResultsRefresh,
}: EvaluateButtonProps) {
  const [isEvaluating, setIsEvaluating] = useState(runStatus === 'evaluating');
  const [error, setError] = useState<string | null>(null);

  const handleEvaluate = useCallback(async () => {
    setIsEvaluating(true);
    setError(null);

    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ runId }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Evaluation failed');
      }

      // Start polling for results
      startPolling();
    } catch (err: any) {
      setError(err.message);
      setIsEvaluating(false);
    }
  }, [runId]);

  const startPolling = useCallback(() => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/evaluate?runId=${encodeURIComponent(runId)}`);

        if (response.ok) {
          const data = await response.json();
          if (data.results && data.results.length > 0) {
            clearInterval(pollInterval);
            setIsEvaluating(false);
            onResultsRefresh?.();
          }
        }
      } catch {
        // Ignore polling errors
      }
    }, 2000);

    // Clean up after 5 minutes max
    setTimeout(() => clearInterval(pollInterval), 300_000);
  }, [runId, onResultsRefresh]);

  // Sync with external runStatus prop
  useEffect(() => {
    if (runStatus === 'evaluating') {
      setIsEvaluating(true);
    }
  }, [runStatus]);

  // Only show when run is completed or evaluating
  if (runStatus !== 'completed' && runStatus !== 'evaluating') {
    return null;
  }

  return (
    <div className="evaluate-button-container">
      <button
        onClick={handleEvaluate}
        disabled={isEvaluating}
        className="evaluate-button"
      >
        {isEvaluating ? 'Evaluating...' : 'Evaluate'}
      </button>

      {error && <span className="error-text">{error}</span>}

      {isEvaluating && (
        <span className="evaluating-status">Evaluating LoRAs...</span>
      )}
    </div>
  );
}
