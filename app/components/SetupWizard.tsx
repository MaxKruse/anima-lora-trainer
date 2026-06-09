'use client';

import { useState, useEffect, useRef } from 'react';

interface SetupStepInfo {
  label: string;
  description: string;
}

const STEP_LABELS: Record<string, SetupStepInfo> = {
  'detect-gpu': { label: 'Detect GPU', description: 'Running nvidia-smi to identify your GPU' },
  'generate-pyproject': { label: 'Generate pyproject.toml', description: 'Writing CUDA-indexed dependency config' },
  'clean-venv': { label: 'Clean old venv', description: 'Removing previous .venv and lock file' },
  'uv-sync': { label: 'Install Python dependencies', description: 'Running uv sync (this may take several minutes)' },
  'verify-pytorch-cuda': { label: 'Verify PyTorch CUDA', description: 'Checking PyTorch CUDA version in venv' },
  'clone-sd-scripts': { label: 'Clone sd-scripts', description: 'Cloning kohya-ss/sd-scripts repository' },
  'done': { label: 'Complete', description: 'All setup steps finished' },
};

interface SetupStepStatus {
  status: 'pending' | 'running' | 'done' | 'error';
  output?: string;
}

interface SetupJobStatus {
  status: 'idle' | 'running' | 'success' | 'error';
  currentStep: string | null;
  steps: Record<string, SetupStepStatus>;
  error?: string;
  gpu?: string;
  series?: string;
  cuda?: string;
  computeCapability?: string;
  pytorchCudaVersion?: string;
  updatedAt: string;
}

interface SetupApiResponse {
  venvReady: boolean;
  sdScriptsReady: boolean;
  setup: SetupJobStatus;
}

export function SetupWizard() {
  const [apiData, setApiData] = useState<SetupApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function startPolling() {
    stopPolling();
    pollTimerRef.current = setInterval(() => {
      fetch('/api/setup')
        .then((res) => res.json())
        .then((data: SetupApiResponse) => setApiData(data))
        .catch(() => {});
    }, 1500);
  }

  function stopPolling() {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }

  // Initial fetch + poll if setup is running
  useEffect(() => {
    function poll() {
      fetch('/api/setup')
        .then((res) => res.json())
        .then((data: SetupApiResponse) => {
          setApiData(data);
          if (data.setup.status === 'running') {
            startPolling();
          }
        })
        .catch(() => {
          // On network error, treat as not ready so the user can attempt setup
          setApiData({
            venvReady: false,
            sdScriptsReady: false,
            setup: {
              status: 'idle',
              currentStep: null,
              steps: {},
              updatedAt: new Date().toISOString(),
            },
          });
        });
    }
    poll();

    return () => stopPolling();
  }, []);

  // Stop polling when setup finishes
  useEffect(() => {
    if (apiData?.setup.status === 'success' || apiData?.setup.status === 'error') {
      stopPolling();
    }
  }, [apiData?.setup.status]);

  async function handleSetup() {
    setLoading(true);
    startPolling();

    try {
      await fetch('/api/setup', { method: 'POST' });
    } catch {
      // Ignore — polling will pick up status
    } finally {
      setLoading(false);
    }
  }

  const isReady = apiData?.venvReady && apiData?.sdScriptsReady;
  const isRunning = apiData?.setup.status === 'running';
  const isSuccess = apiData?.setup.status === 'success';
  const isError = apiData?.setup.status === 'error';
  const steps = apiData?.setup.steps ?? {};

  // Still loading initial state
  if (!apiData) {
    return (
      <div className="max-w-lg mx-auto p-6">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6">Setup</h1>
        <p className="text-slate-500 dark:text-slate-400">Checking environment...</p>
      </div>
    );
  }

  return (
    <div className="max-w-lg mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Setup</h1>

      {/* Readiness badges */}
      <div className="flex gap-3">
        <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${
          apiData.venvReady
            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
            : 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
        }`}>
          <span className={`w-2 h-2 rounded-full ${apiData.venvReady ? 'bg-green-500' : 'bg-amber-500'}`} />
          Python venv {apiData.venvReady ? 'ready' : 'missing'}
        </span>
        <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${
          apiData.sdScriptsReady
            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
            : 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
        }`}>
          <span className={`w-2 h-2 rounded-full ${apiData.sdScriptsReady ? 'bg-green-500' : 'bg-amber-500'}`} />
          sd-scripts {apiData.sdScriptsReady ? 'ready' : 'missing'}
        </span>
      </div>

      {/* Success state */}
      {isReady && !isRunning && (
        <div className="space-y-4">
          <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
            <p className="font-semibold text-green-800 dark:text-green-400">Environment is ready</p>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              Python venv and sd-scripts are set up and ready to use.
            </p>
            {(apiData.setup.gpu || apiData.setup.cuda || apiData.setup.computeCapability || apiData.setup.pytorchCudaVersion) && (
              <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                {apiData.setup.gpu && (
                  <>
                    <dt className="text-slate-500 dark:text-slate-400">GPU:</dt>
                    <dd className="font-medium text-slate-900 dark:text-slate-100">{apiData.setup.gpu}</dd>
                  </>
                )}
                {apiData.setup.computeCapability && (
                  <>
                    <dt className="text-slate-500 dark:text-slate-400">Compute capability:</dt>
                    <dd className="font-medium text-slate-900 dark:text-slate-100">{apiData.setup.computeCapability}</dd>
                  </>
                )}
                {apiData.setup.pytorchCudaVersion && (
                  <>
                    <dt className="text-slate-500 dark:text-slate-400">PyTorch CUDA:</dt>
                    <dd className="font-medium text-slate-900 dark:text-slate-100">{apiData.setup.pytorchCudaVersion}</dd>
                  </>
                )}
                {apiData.setup.cuda && (
                  <>
                    <dt className="text-slate-500 dark:text-slate-400">PyTorch index:</dt>
                    <dd className="font-medium text-slate-900 dark:text-slate-100">{apiData.setup.cuda}</dd>
                  </>
                )}
              </dl>
            )}
          </div>

          <button
            onClick={handleSetup}
            disabled={loading || isRunning}
            className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-md hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 transition-colors text-sm"
          >
            {isRunning ? 'Re-installing...' : 'Re-install environment'}
          </button>
        </div>
      )}

      {/* Not ready yet — initial setup */}
      {!isReady && !isRunning && !isError && (
        <div className="space-y-4">
          <div className="p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
            <p className="font-semibold text-amber-800 dark:text-amber-400">Setup required</p>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              {(!apiData.venvReady && !apiData.sdScriptsReady)
                ? 'Python venv and sd-scripts need to be set up.'
                : (!apiData.venvReady
                  ? 'Python venv needs to be set up.'
                  : 'sd-scripts repository needs to be cloned.')
              }
            </p>
          </div>

          <button
            onClick={handleSetup}
            disabled={loading || isRunning}
            className="px-4 py-2 bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white rounded-md hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50 transition-colors"
          >
            {isRunning ? 'Setting up...' : 'Setup Environment'}
          </button>
        </div>
      )}

      {/* Progress steps — only shown during active setup/reinstall */}
      {isRunning && (
        <div className="space-y-2">
          {Object.keys(STEP_LABELS).map((stepKey) => {
            const step = steps[stepKey] as SetupStepInfo | SetupStepStatus | undefined;
            const stepStatus = typeof step === 'object' && 'status' in step ? step as SetupStepStatus : { status: 'pending' };
            const info = STEP_LABELS[stepKey];
            const isCurrent = apiData.setup.currentStep === stepKey;

            return (
              <div
                key={stepKey}
                className={`flex items-start gap-3 p-3 rounded-lg border transition-colors ${
                  isCurrent
                    ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                    : stepStatus.status === 'done'
                    ? 'bg-green-50 dark:bg-green-900/10 border-green-200 dark:border-green-900/50'
                    : stepStatus.status === 'error'
                    ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                    : 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700'
                }`}
              >
                {/* Status icon */}
                <div className="mt-0.5 flex-shrink-0">
                  {stepStatus.status === 'running' && (
                    <svg className="w-5 h-5 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                  )}
                  {stepStatus.status === 'done' && (
                    <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                  {stepStatus.status === 'error' && (
                    <svg className="w-5 h-5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  )}
                  {stepStatus.status === 'pending' && (
                    <div className="w-5 h-5 rounded-full border-2 border-slate-300 dark:border-slate-600" />
                  )}
                </div>

                {/* Step info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className={`text-sm font-medium ${
                      isCurrent
                        ? 'text-blue-700 dark:text-blue-300'
                        : stepStatus.status === 'done'
                        ? 'text-green-700 dark:text-green-400'
                        : stepStatus.status === 'error'
                        ? 'text-red-700 dark:text-red-400'
                        : 'text-slate-500 dark:text-slate-400'
                    }`}>
                      {info.label}
                    </p>
                    {isCurrent && (
                      <span className="text-xs text-blue-500 dark:text-blue-400 animate-pulse">in progress</span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400">{info.description}</p>
                  {stepStatus.output && (stepStatus.status === 'done' || stepStatus.status === 'error') && (
                    <pre className="mt-1 text-xs text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 rounded p-2 max-h-32 overflow-auto whitespace-pre-wrap [&::-webkit-scrollbar]:h-1.5 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:bg-slate-400 dark:[&::-webkit-scrollbar-thumb]:bg-slate-500 [&::-webkit-scrollbar-thumb]:rounded-full">
                      {stepStatus.output}
                    </pre>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Error state */}
      {isError && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="font-semibold text-red-700 dark:text-red-400">Setup failed</p>
          {apiData.setup.error && (
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">{apiData.setup.error}</p>
          )}
          <button
            onClick={handleSetup}
            disabled={loading || isRunning}
            className="mt-3 px-3 py-1.5 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm disabled:opacity-50 transition-colors"
          >
            Retry
          </button>
        </div>
      )}
    </div>
  );
}
