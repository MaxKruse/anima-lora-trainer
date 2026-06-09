'use client';

import { useState } from 'react';

interface SetupResult {
  gpu: string;
  series: string;
  cuda: string;
  cudaVersion?: string;
  pyprojectPath?: string;
  status: 'ok' | 'error';
  error?: string;
}

export function SetupWizard() {
  const [result, setResult] = useState<SetupResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleDetect() {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/setup', { method: 'POST' });
      const data: SetupResult = await res.json();

      if (!res.ok || data.status === 'error') {
        setError(data.error || 'Setup failed');
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-lg mx-auto p-6">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6">Setup Wizard</h1>

      {!result && !error && (
        <button
          onClick={handleDetect}
          disabled={loading}
          className="px-4 py-2 bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white rounded hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Detecting...' : 'Detect GPU'}
        </button>
      )}

      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-red-700 dark:text-red-400">
          <p className="font-semibold">Error</p>
          <p>{error}</p>
          <button
            onClick={handleDetect}
            disabled={loading}
            className="mt-2 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm disabled:opacity-50"
          >
            Retry
          </button>
        </div>
      )}

      {result && result.status === 'ok' && (
        <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
          <p className="font-semibold text-green-800 dark:text-green-400 mb-2">Environment ready</p>
          <dl className="space-y-1 text-sm">
            <div>
              <dt className="text-slate-600 dark:text-slate-400">GPU:</dt>
              <dd className="font-medium text-slate-900 dark:text-slate-100">{result.gpu}</dd>
            </div>
            <div>
              <dt className="text-slate-600 dark:text-slate-400">Architecture:</dt>
              <dd className="font-medium capitalize text-slate-900 dark:text-slate-100">{result.series}</dd>
            </div>
            <div>
              <dt className="text-slate-600 dark:text-slate-400">CUDA Toolkit:</dt>
              <dd className="font-medium text-slate-900 dark:text-slate-100">{result.cudaVersion || 'Unknown'}</dd>
            </div>
            <div>
              <dt className="text-slate-600 dark:text-slate-400">PyTorch Index:</dt>
              <dd className="font-medium text-slate-900 dark:text-slate-100">{result.cuda}</dd>
            </div>
            {result.pyprojectPath && (
              <div>
                <dt className="text-slate-600 dark:text-slate-400">Config:</dt>
                <dd className="font-medium text-xs break-all text-slate-900 dark:text-slate-100">{result.pyprojectPath}</dd>
              </div>
            )}
          </dl>
        </div>
      )}
    </div>
  );
}
