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
    <div className="flex flex-col gap-2">
      <button
        onClick={handleEvaluate}
        disabled={isEvaluating}
        className="px-4 py-2 text-sm bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white rounded-md hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isEvaluating ? 'Evaluating...' : 'Evaluate'}
      </button>

      {error && <span className="text-red-600 dark:text-red-400 text-sm">{error}</span>}

      {isEvaluating && (
        <span className="text-slate-500 dark:text-slate-400 text-sm">Evaluating LoRAs...</span>
      )}
    </div>
  );
}
