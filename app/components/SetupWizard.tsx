'use client';

import { useState } from 'react';

interface SetupResult {
  gpu: string;
  series: string;
  cuda: string;
  pyprojectPath?: string;
  status: 'ok' | 'error';
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
      <h1 className="text-2xl font-bold mb-6">Setup Wizard</h1>

      {!result && !error && (
        <button
          onClick={handleDetect}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Detecting...' : 'Detect GPU'}
        </button>
      )}

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded text-red-700">
          <p className="font-semibold">Error</p>
          <p>{error}</p>
          <button
            onClick={handleDetect}
            disabled={loading}
            className="mt-2 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
          >
            Retry
          </button>
        </div>
      )}

      {result && result.status === 'ok' && (
        <div className="p-4 bg-green-50 border border-green-200 rounded">
          <p className="font-semibold text-green-800 mb-2">Environment ready</p>
          <dl className="space-y-1 text-sm">
            <div>
              <dt className="text-gray-600">GPU:</dt>
              <dd className="font-medium">{result.gpu}</dd>
            </div>
            <div>
              <dt className="text-gray-600">Architecture:</dt>
              <dd className="font-medium capitalize">{result.series}</dd>
            </div>
            <div>
              <dt className="text-gray-600">CUDA Version:</dt>
              <dd className="font-medium">{result.cuda}</dd>
            </div>
            {result.pyprojectPath && (
              <div>
                <dt className="text-gray-600">Config:</dt>
                <dd className="font-medium text-xs break-all">{result.pyprojectPath}</dd>
              </div>
            )}
          </dl>
        </div>
      )}
    </div>
  );
}
