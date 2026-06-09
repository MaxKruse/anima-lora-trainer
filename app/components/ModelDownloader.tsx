'use client';

import { useState, useEffect, useCallback } from 'react';

interface ModelStatus {
  name: string;
  expectedSizeBytes: number;
  status: 'pending' | 'downloading' | 'downloaded';
  progress: number;
  sizeBytes?: number;
  canAbort?: boolean;
  error?: string;
  cachePath?: string;
}

const CIRCLE_SIZE = 40;
const STROKE_WIDTH = 3;
const RADIUS = (CIRCLE_SIZE - STROKE_WIDTH) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function CircularProgress({ progress, canAbort, onAbort }: {
  progress: number;
  canAbort: boolean;
  onAbort: () => void;
}) {
  const offset = CIRCUMFERENCE - (progress / 100) * CIRCUMFERENCE;

  return (
    <div className="relative inline-block">
      <svg
        width={CIRCLE_SIZE}
        height={CIRCLE_SIZE}
        role="progressbar"
        aria-valuenow={progress}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        {/* Background track */}
        <circle
          cx={CIRCLE_SIZE / 2}
          cy={CIRCLE_SIZE / 2}
          r={RADIUS}
          fill="none"
          stroke="currentColor"
          strokeWidth={STROKE_WIDTH}
          className="text-slate-200 dark:text-slate-700"
        />
        {/* Filling arc */}
        <circle
          cx={CIRCLE_SIZE / 2}
          cy={CIRCLE_SIZE / 2}
          r={RADIUS}
          fill="none"
          stroke="currentColor"
          strokeWidth={STROKE_WIDTH}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform={`rotate(-90 ${CIRCLE_SIZE / 2} ${CIRCLE_SIZE / 2})`}
          className="text-slate-900 dark:text-slate-100 transition-all duration-300"
        />
      </svg>
      {/* Center percentage text */}
      <span className="absolute inset-0 flex items-center justify-center text-[10px] font-medium text-slate-500 dark:text-slate-400">
        {progress}%
      </span>
      {/* Abort button overlay — visible on hover when abortable */}
      {canAbort && (
        <button
          onClick={onAbort}
          className="absolute inset-0 w-full h-full flex items-center justify-center bg-red-500 rounded opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
          title="Abort download"
          aria-label="Abort download"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
          >
            <rect x="2" y="2" width="12" height="12" fill="white" rx="1" />
          </svg>
        </button>
      )}
    </div>
  );
}

export function ModelDownloader() {
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      if (res.ok && data.models) {
        setModels(data.models);
      }
    } catch {
      // Ignore fetch errors during polling
    }
  }, []);

  // Initial status fetch + polling during downloads
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    const hasDownloading = models.some(m => m.status === 'downloading');
    if (!hasDownloading) return;

    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [models, fetchStatus]);

  async function handleDownload(modelName: string) {
    setLoading(true);
    setErrors(prev => ({ ...prev, [modelName]: '' }));

    try {
      const res = await fetch('/api/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelName }),
      });

      const data = await res.json();

      if (!res.ok) {
        setErrors(prev => ({ ...prev, [modelName]: data.error || 'Download failed' }));
      } else {
        // Update status to downloading
        setModels(prev =>
          prev.map(m =>
            m.name === modelName ? { ...m, status: 'downloading' as const, progress: 0, canAbort: true } : m
          )
        );
      }
    } catch (err: any) {
      setErrors(prev => ({ ...prev, [modelName]: err.message || 'Network error' }));
    } finally {
      setLoading(false);
    }
  }

  async function handleAbort(modelName: string) {
    try {
      const res = await fetch(`/api/models?modelName=${encodeURIComponent(modelName)}`, {
        method: 'DELETE',
      });

      if (!res.ok) {
        const data = await res.json();
        setErrors(prev => ({ ...prev, [modelName]: data.error || 'Abort failed' }));
      }
    } catch {
      // Ignore abort errors — polling will update status
    }
  }

  function formatSize(bytes: number): string {
    if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
    if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
    return `${(bytes / 1_000).toFixed(0)} KB`;
  }

  return (
    <div className="max-w-lg mx-auto p-6">
      <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4">Model Downloads</h2>

      {models.length === 0 && (
        <p className="text-slate-500 dark:text-slate-400">Loading model list...</p>
      )}

      <div className="space-y-3">
        {models.map((model) => (
          <div
            key={model.name}
            className="group p-4 border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800"
          >
            <div className="flex items-center justify-between">
              <div>
                <span className="font-medium text-slate-900 dark:text-slate-100">{model.name}</span>
                <span className="text-sm text-slate-500 dark:text-slate-400 ml-2">
                  ({formatSize(model.expectedSizeBytes)})
                </span>
              </div>

              {model.status === 'downloaded' && (
                <span className="text-green-600 dark:text-green-400 text-lg font-bold" title="Completed">✓</span>
              )}

              {model.status === 'downloading' && (
                <CircularProgress
                  progress={model.progress}
                  canAbort={!!model.canAbort && !model.error}
                  onAbort={() => handleAbort(model.name)}
                />
              )}

              {(errors[model.name] || model.error) && model.status !== 'downloading' && (
                <span className="text-red-600 dark:text-red-400 text-sm">
                  {model.error || errors[model.name]}
                </span>
              )}

              {model.error && model.status === 'downloading' && (
                <span className="text-red-600 dark:text-red-400 text-sm">{model.error}</span>
              )}
            </div>

            {model.status === 'pending' && (
              <div className="mt-2">
                <button
                  onClick={() => {
                    // Clear any previous error when starting a new download
                    setErrors(prev => ({ ...prev, [model.name]: '' }));
                    handleDownload(model.name);
                  }}
                  disabled={loading}
                  className="px-3 py-1 bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white text-sm rounded hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50 transition-colors"
                >
                  Download
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
