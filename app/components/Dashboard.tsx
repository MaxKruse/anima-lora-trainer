'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { DirectoryPicker } from './DirectoryPicker';
import { TrainTabs } from './TrainTabs';
import { MatrixToggle } from './MatrixToggle';
import { JobList } from './JobList';
import { SetupWizard } from './SetupWizard';
import { ModelDownloader } from './ModelDownloader';
import { ResultsDashboard } from './ResultsDashboard';
import { type TrainingParams } from '../lib/training-schema';

interface AppConfig {
  trainingImagesDir: string;
  outputDir: string;
  modelsDir?: string;
  sdCliPath?: string;
  sdScriptsPath?: string;
}

interface DatasetDir {
  name: string;
  path: string;
}

type DashboardSection = 'setup' | 'models' | 'train' | 'jobs' | 'results';

interface SetupReadiness {
  venvReady: boolean;
  sdScriptsReady: boolean;
}

export function Dashboard() {
  const [activeSection, setActiveSection] = useState<DashboardSection>('train');
  const [config, setConfig] = useState<AppConfig>({
    trainingImagesDir: '',
    outputDir: '',
  });
  const [savedConfig, setSavedConfig] = useState<AppConfig | null>(null);
  const [configErrors, setConfigErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  // Dataset browser state
  const [datasets, setDatasets] = useState<DatasetDir[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Detect unsaved changes
  const hasUnsaved = savedConfig !== null && (
    config.trainingImagesDir !== savedConfig.trainingImagesDir ||
    config.outputDir !== savedConfig.outputDir
  );
  const [trainingResponse, setTrainingResponse] = useState<string | null>(null);
  const [matrixMode, setMatrixMode] = useState<'single' | 'matrix'>('single');
  const [permutationCount, setPermutationCount] = useState(0);

  // Setup readiness gate
  const [setupReadiness, setSetupReadiness] = useState<SetupReadiness | null>(null);
  const isSetupComplete = setupReadiness?.venvReady && setupReadiness?.sdScriptsReady;
  const isSetupChecking = setupReadiness === null;

  // Check setup readiness on mount
  useEffect(() => {
    fetch('/api/setup')
      .then((res) => res.json() as Promise<{ venvReady: boolean; sdScriptsReady: boolean }>)
      .then((data) => {
        setSetupReadiness({ venvReady: data.venvReady, sdScriptsReady: data.sdScriptsReady });
        // If setup is incomplete, force-navigate to Setup section
        if (!data.venvReady || !data.sdScriptsReady) {
          setActiveSection('setup');
        }
      })
      .catch(() => {
        // Network error — treat as not ready and show Setup
        setSetupReadiness({ venvReady: false, sdScriptsReady: false });
        setActiveSection('setup');
      });
  }, []);

  // Re-check readiness when leaving Setup section (in case setup just completed)
  useEffect(() => {
    if (activeSection !== 'setup') return;
    const timer = setInterval(() => {
      fetch('/api/setup')
        .then((res) => res.json() as Promise<{ venvReady: boolean; sdScriptsReady: boolean; setup: { status: string } }>)
        .then((data) => {
          setSetupReadiness({ venvReady: data.venvReady, sdScriptsReady: data.sdScriptsReady });
          // If setup just completed and we're still on Setup, stop polling
          if (data.venvReady && data.sdScriptsReady && data.setup.status === 'success') {
            clearInterval(timer);
          }
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(timer);
  }, [activeSection]);

  // Load config on mount
  useEffect(() => {
    fetch('/api/config')
      .then((res) => res.json())
      .then((data) => {
        if (data.config) {
          setConfig(data.config);
          setSavedConfig(data.config);
        }
      })
      .catch(() => {
        // Config load failed — use defaults
      });
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Load datasets
  const loadDatasets = useCallback(async () => {
    if (datasets.length > 0) {
      setShowDropdown((prev) => !prev);
      return;
    }
    setDatasetsLoading(true);
    setDatasetsError(null);
    try {
      const res = await fetch('/api/datasets');
      const data = await res.json();
      if (res.ok) {
        setDatasets(data.directories || []);
        setShowDropdown(true);
        if (data.directories?.length === 0) {
          setDatasetsError('No dataset directories found in datasets/');
        }
      } else {
        setDatasetsError(data.error || 'Failed to load datasets');
      }
    } catch {
      setDatasetsError('Failed to connect to server');
    } finally {
      setDatasetsLoading(false);
    }
  }, [datasets.length]);

  const saveConfig = useCallback(async (updates: Partial<AppConfig>) => {
    setSaving(true);
    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updates }),
      });
      if (res.ok) {
        const data = await res.json();
        setConfig(data.config);
        setSavedConfig(data.config);
      }
    } catch {
      // Ignore save errors
    } finally {
      setSaving(false);
    }
  }, []);

  const handleConfigChange = useCallback(
    (key: keyof AppConfig, value: string) => {
      setConfig((prev) => ({ ...prev, [key]: value }));
      // Clear error for this field
      if (configErrors[key]) {
        setConfigErrors((prev) => {
          const next = { ...prev };
          delete next[key];
          return next;
        });
      }
    },
    [configErrors]
  );

  const selectDataset = useCallback((datasetPath: string) => {
    handleConfigChange('trainingImagesDir', datasetPath);
    setShowDropdown(false);
  }, [handleConfigChange]);

  const handleMatrixModeChange = useCallback(
    (mode: 'single' | 'matrix') => {
      setMatrixMode(mode);
      setPermutationCount(0);
    },
    []
  );

  const handleTrainingSubmit = useCallback(
    async (params: TrainingParams) => {
      setTrainingResponse(null);

      try {
        const res = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params),
        });

        const data = await res.json();

        if (!res.ok) {
          setTrainingResponse(`Error: ${data.error || 'Training failed to start'}`);
          return;
        }

        setTrainingResponse(`Job ${data.jobId} started successfully!`);
        setActiveSection('jobs');
      } catch (err: any) {
        setTrainingResponse(`Error: ${err.message || 'Network error'}`);
      }
    },
    []
  );

  const handleMatrixTrainingSubmit = useCallback(
    async (paramRanges: Record<string, string>, baseParams: Record<string, any>) => {
      setTrainingResponse(null);

      try {
        const res = await fetch('/api/train/matrix', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ paramRanges, baseParams }),
        });

        const data = await res.json();

        if (!res.ok) {
          setTrainingResponse(`Error: ${data.error || 'Matrix training failed to start'}`);
          return;
        }

        setTrainingResponse(`Matrix job ${data.jobId} started with ${data.permutationCount} permutations!`);
        setActiveSection('jobs');
      } catch (err: any) {
        setTrainingResponse(`Error: ${err.message || 'Network error'}`);
      }
    },
    []
  );

  const sections: { key: DashboardSection; label: string }[] = [
    { key: 'setup', label: 'Setup' },
    { key: 'models', label: 'Models' },
    { key: 'train', label: 'Train' },
    { key: 'jobs', label: 'Jobs' },
    { key: 'results', label: 'Results' },
  ];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                LoRA Matrix Trainer
              </h1>
              <span className="px-2 py-0.5 text-xs font-medium bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-full">
                Beta
              </span>
            </div>

            {/* Section navigation */}
            <nav className="flex gap-1">
              {sections.map((section) => {
                const isLocked = !isSetupComplete && section.key !== 'setup';
                const isActive = activeSection === section.key;

                return (
                  <button
                    key={section.key}
                    onClick={() => {
                      if (!isLocked) setActiveSection(section.key);
                    }}
                    disabled={isLocked}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                      isLocked
                        ? 'text-slate-300 dark:text-slate-600 cursor-not-allowed'
                        : isActive
                        ? 'bg-slate-200 dark:bg-slate-700 text-slate-900 dark:text-slate-100'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                    }`}
                    title={isLocked ? 'Complete setup to unlock' : undefined}
                  >
                    {section.label}
                    {isLocked && (
                      <svg className="inline-block w-3.5 h-3.5 ml-1.5 -mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                      </svg>
                    )}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Brief loading state while setup readiness is being checked */}
        {isSetupChecking && (
          <div className="max-w-lg mx-auto p-6">
            <p className="text-slate-500 dark:text-slate-400">Checking environment...</p>
          </div>
        )}

        {!isSetupChecking && activeSection === 'setup' && <SetupWizard />}

        {activeSection === 'models' && <ModelDownloader />}

        {activeSection === 'train' && (
          <div className="space-y-8">
            {/* Directory Configuration */}
            <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">
                Directories
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                Configure the input and output directories for training. Paths are saved
                automatically and persist between sessions.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <DirectoryPicker
                  id="trainingImagesDir"
                  label="Training Images Directory"
                  value={config.trainingImagesDir}
                  onChange={(v) => handleConfigChange('trainingImagesDir', v)}
                  placeholder="/path/to/training/images"
                  hint="Folder containing training images and .txt caption files"
                  error={configErrors.trainingImagesDir}
                />

                <DirectoryPicker
                  id="outputDir"
                  label="Output Directory"
                  value={config.outputDir}
                  onChange={(v) => handleConfigChange('outputDir', v)}
                  placeholder="outputs"
                  hint="Directory name or path — created automatically if it doesn't exist"
                  error={configErrors.outputDir}
                  autoVerify={false}
                />
              </div>

              {/* Browse Datasets */}
              <div ref={dropdownRef} className="mt-4">
                <button
                  type="button"
                  onClick={loadDatasets}
                  disabled={datasetsLoading}
                  className="w-full px-4 py-2 text-sm bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-600 rounded-md hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                  title="Browse dataset directories on server"
                >
                  {datasetsLoading ? (
                    <span className="flex items-center justify-center gap-1">
                      <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Loading
                    </span>
                  ) : showDropdown ? (
                    '▲ Hide Datasets'
                  ) : (
                    'Browse Datasets'
                  )}
                </button>

                {datasetsError && (
                  <p className="mt-1 text-xs text-yellow-600 dark:text-yellow-400">{datasetsError}</p>
                )}

                {/* Dropdown list of datasets */}
                {showDropdown && datasets.length > 0 && (
                  <div className="absolute top-full left-0 right-0 mt-1 max-h-60 overflow-y-auto bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md shadow-lg z-20">
                    {datasets.map((ds) => (
                      <button
                        key={ds.path}
                        type="button"
                        onClick={() => selectDataset(ds.path)}
                        className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors border-b border-slate-100 dark:border-slate-700 last:border-b-0 ${
                          config.trainingImagesDir === ds.path
                            ? 'bg-slate-100 dark:bg-slate-700 font-medium'
                            : ''
                        }`}
                      >
                        <div className="font-medium text-slate-900 dark:text-slate-100 truncate">
                          {ds.name}
                        </div>
                        <div className="text-xs text-slate-400 dark:text-slate-500 font-mono truncate">
                          {ds.path}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Save button */}
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() =>
                    saveConfig({
                      trainingImagesDir: config.trainingImagesDir,
                      outputDir: config.outputDir,
                    })
                  }
                  disabled={saving}
                  className={`w-full px-4 py-2 text-sm rounded-md disabled:opacity-50 transition-colors ${
                    hasUnsaved
                      ? 'bg-amber-600 hover:bg-amber-700 dark:bg-amber-500 dark:hover:bg-amber-600 text-white animate-pulse'
                      : 'bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white hover:bg-slate-800 dark:hover:bg-slate-200'
                  }`}
                >
                  {saving ? 'Saving...' : hasUnsaved ? 'Save Directories (unsaved changes)' : 'Save Directories'}
                </button>
              </div>
            </section>

            {/* Matrix Mode Toggle */}
            <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                  Training Configuration
                </h2>
                <MatrixToggle
                  mode={matrixMode}
                  onChange={handleMatrixModeChange}
                  permutationCount={permutationCount}
                />
              </div>

              <TrainTabs
                trainingImagesPath={config.trainingImagesDir}
                onSubmit={handleTrainingSubmit}
                onMatrixSubmit={handleMatrixTrainingSubmit}
                onPermutationCountChange={setPermutationCount}
                matrixMode={matrixMode === 'matrix'}
              />

              {trainingResponse && (
                <div
                  className={`mt-4 p-4 rounded-lg text-sm ${
                    trainingResponse.startsWith('Error')
                      ? 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400'
                      : 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                  }`}
                >
                  {trainingResponse}
                </div>
              )}
            </section>
          </div>
        )}

        {activeSection === 'jobs' && <JobList />}

        {activeSection === 'results' && <ResultsDashboard />}
      </main>
    </div>
  );
}
