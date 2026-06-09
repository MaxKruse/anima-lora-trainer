'use client';

import { useState, useEffect, useCallback } from 'react';
import { DirectoryPicker } from './DirectoryPicker';
import { AnimaTab } from './AnimaTab';
import { MatrixToggle } from './MatrixToggle';
import { JobList } from './JobList';
import { SetupWizard } from './SetupWizard';
import { ModelDownloader } from './ModelDownloader';
import { type TrainingParams } from '../lib/training-schema';

interface AppConfig {
  trainingImagesDir: string;
  outputDir: string;
  modelsDir?: string;
  sdCliPath?: string;
  sdScriptsPath?: string;
}

type DashboardSection = 'setup' | 'models' | 'train' | 'jobs';

export function Dashboard() {
  const [activeSection, setActiveSection] = useState<DashboardSection>('train');
  const [config, setConfig] = useState<AppConfig>({
    trainingImagesDir: '',
    outputDir: '',
  });
  const [configErrors, setConfigErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [trainingResponse, setTrainingResponse] = useState<string | null>(null);
  const [matrixMode, setMatrixMode] = useState<'single' | 'matrix'>('single');

  // Load config on mount
  useEffect(() => {
    fetch('/api/config')
      .then((res) => res.json())
      .then((data) => {
        if (data.config) {
          setConfig(data.config);
        }
      })
      .catch(() => {
        // Config load failed — use defaults
      });
  }, []);

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

  const handleMatrixModeChange = useCallback(
    (mode: 'single' | 'matrix') => {
      setMatrixMode(mode);
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

  const sections: { key: DashboardSection; label: string }[] = [
    { key: 'setup', label: 'Setup' },
    { key: 'models', label: 'Models' },
    { key: 'train', label: 'Train' },
    { key: 'jobs', label: 'Jobs' },
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
              {sections.map((section) => (
                <button
                  key={section.key}
                  onClick={() => setActiveSection(section.key)}
                  className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                    activeSection === section.key
                      ? 'bg-slate-200 dark:bg-slate-700 text-slate-900 dark:text-slate-100'
                      : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                  }`}
                >
                  {section.label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeSection === 'setup' && <SetupWizard />}

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

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl">
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
                  placeholder="/path/to/outputs"
                  hint="Where trained LoRA files will be saved"
                  error={configErrors.outputDir}
                />
              </div>

              {/* Save button */}
              <div className="mt-4">
                <button
                  type="button"
                  onClick={() =>
                    saveConfig({
                      trainingImagesDir: config.trainingImagesDir,
                      outputDir: config.outputDir,
                    })
                  }
                  disabled={saving}
                  className="px-4 py-2 text-sm bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white rounded-md hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50 transition-colors"
                >
                  {saving ? 'Saving...' : 'Save Directories'}
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
                />
              </div>

              <AnimaTab
                trainingImagesPath={config.trainingImagesDir}
                onSubmit={handleTrainingSubmit}
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
      </main>
    </div>
  );
}
