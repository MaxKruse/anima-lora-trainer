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

  const sections: { key: DashboardSection; label: string; icon: string }[] = [
    { key: 'setup', label: 'Setup', icon: '⚙️' },
    { key: 'models', label: 'Models', icon: '📦' },
    { key: 'train', label: 'Train', icon: '🚀' },
    { key: 'jobs', label: 'Jobs', icon: '📋' },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      {/* Header */}
      <header className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                LoRA Matrix Trainer
              </h1>
              <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-full">
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
                      ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'
                      : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800/50'
                  }`}
                >
                  <span className="mr-1.5">{section.icon}</span>
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
            <section className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Directories
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
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
                  className="px-4 py-2 text-sm bg-gray-900 dark:bg-white dark:text-gray-900 text-white rounded-md hover:bg-gray-800 dark:hover:bg-gray-100 disabled:opacity-50 transition-colors"
                >
                  {saving ? 'Saving...' : 'Save Directories'}
                </button>
              </div>
            </section>

            {/* Matrix Mode Toggle */}
            <section className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
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
