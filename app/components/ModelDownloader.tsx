'use client';

import { useState, useEffect, useCallback } from 'react';

interface ModelStatus {
  name: string;
  localPath: string;
  expectedSizeBytes: number;
  status: 'pending' | 'downloading' | 'downloaded';
  progress: number;
  sizeBytes?: number;
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
            m.name === modelName ? { ...m, status: 'downloading' as const, progress: 0 } : m
          )
        );
      }
    } catch (err: any) {
      setErrors(prev => ({ ...prev, [modelName]: err.message || 'Network error' }));
    } finally {
      setLoading(false);
    }
  }

  function formatSize(bytes: number): string {
    if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
    if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
    return `${(bytes / 1_000).toFixed(0)} KB`;
  }

  return (
    <div className="max-w-lg mx-auto p-6">
      <h2 className="text-xl font-bold mb-4">Model Downloads</h2>

      {models.length === 0 && (
        <p className="text-gray-500">Loading model list...</p>
      )}

      <div className="space-y-3">
        {models.map((model) => (
          <div
            key={model.name}
            className="p-4 border rounded-lg bg-white"
          >
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="font-medium">{model.name}</span>
                <span className="text-sm text-gray-500 ml-2">
                  ({formatSize(model.expectedSizeBytes)})
                </span>
              </div>

              {model.status === 'downloaded' && (
                <span className="text-green-600 text-lg" title="Completed">✓</span>
              )}

              {errors[model.name] && (
                <span className="text-red-600 text-sm">{errors[model.name]}</span>
              )}
            </div>

            {model.status === 'downloading' && (
              <div className="mb-2">
                <div
                  className="w-full bg-gray-200 rounded-full h-2"
                  role="progressbar"
                  aria-valuenow={model.progress}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all"
                    style={{ width: `${model.progress}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500">{model.progress}%</span>
              </div>
            )}

            {model.status === 'pending' && (
              <button
                onClick={() => handleDownload(model.name)}
                disabled={loading}
                className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
              >
                Download
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
