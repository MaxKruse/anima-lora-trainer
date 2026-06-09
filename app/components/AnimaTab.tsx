'use client';

import { useState } from 'react';
import { trainingSchema, type TrainingParams } from '../lib/training-schema';
import { MultiSelectDropdown } from './MultiSelectDropdown';

interface AnimaTabProps {
  onSubmit: (params: TrainingParams) => void;
  /**
   * When provided, the training images path is pre-filled and the field
   * is hidden (managed by the parent directory picker).
   */
  trainingImagesPath?: string;
  /**
   * When true, render multi-select dropdowns for parameter inputs
   * to support matrix training (multiple values per parameter).
   */
  matrixMode?: boolean;
}

const OPTIMIZERS = ['AdamW8Bit', 'AdamW', 'Prodigy', 'Lion', 'Adafactor'];
const SCHEDULERS = ['constant', 'cosine', 'linear', 'constant_with_warmup', 'cosine_with_restarts'];
const MIXED_PRECISIONS = ['fp16', 'bf16', 'no'];
const TIMESTEP_SAMPLINGS = ['sigma', 'uniform', 'sigmoid', 'shift', 'flux_shift'];

const DEFAULT_NETWORK_DIMS = ['8', '16', '32', '64', '128', '256'];
const DEFAULT_NETWORK_ALPHAS = ['4', '8', '16', '32', '64'];
const DEFAULT_LEARNING_RATES = ['0.0001', '0.00005', '0.00001', '0.001'];
const DEFAULT_BATCH_SIZES = ['1', '2', '4', '8', '16'];
const DEFAULT_EPOCHS = ['1', '5', '10', '20', '50'];

const DEFAULT_PARAMS: Omit<TrainingParams, 'trainingImages' | 'loraName'> = {
  networkDim: 32,
  networkAlpha: 16,
  learningRate: 1e-4,
  batchSize: 1,
  epochs: 10,
  optimizer: 'AdamW8Bit',
  scheduler: 'cosine',
  mixedPrecision: 'bf16',
  timestepSampling: 'sigmoid',
  gradientCheckpointing: true,
  cacheLatents: true,
  cacheTextEncoder: true,
};

// Default matrix values (single value arrays for initial state)
const DEFAULT_MATRIX_VALUES: Record<string, string[]> = {
  networkDim: ['32'],
  networkAlpha: ['16'],
  learningRate: ['0.0001'],
  batchSize: ['1'],
  epochs: ['10'],
  optimizer: ['AdamW8Bit'],
  scheduler: ['cosine'],
  mixedPrecision: ['bf16'],
  timestepSampling: ['sigmoid'],
};

export function AnimaTab({ onSubmit, trainingImagesPath, matrixMode = false }: AnimaTabProps) {
  const isManagedExternally = trainingImagesPath !== undefined;
  const [params, setParams] = useState({
    ...DEFAULT_PARAMS,
    trainingImages: trainingImagesPath || '',
    loraName: '',
  });
  // Matrix mode: multi-value params stored as string arrays
  const [matrixValues, setMatrixValues] = useState<Record<string, string[]>>(DEFAULT_MATRIX_VALUES);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  function updateParam<K extends keyof typeof params>(key: K, value: (typeof params)[K]) {
    setParams((prev) => ({ ...prev, [key]: value }));
    if (errors[key]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }

  function updateMatrixParam(key: string, values: string[]) {
    setMatrixValues((prev) => ({ ...prev, [key]: values }));
    if (errors[key]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }

  function validate(): boolean {
    const newErrors: Record<string, string> = {};

    if (!isManagedExternally && !params.trainingImages.trim()) {
      newErrors.trainingImages = 'Training images path is required';
    }
    if (!params.loraName.trim()) {
      newErrors.loraName = 'LoRA name is required';
    }

    if (matrixMode) {
      // Validate matrix values have at least one entry for each param
      for (const [key, values] of Object.entries(matrixValues)) {
        if (values.length === 0) {
          newErrors[key] = `At least one value is required for ${key}`;
        }
      }
    } else {
      if (params.networkDim < 1) {
        newErrors.networkDim = 'Network dim must be at least 1';
      }
      if (params.epochs < 1) {
        newErrors.epochs = 'Epochs must be at least 1';
      }
      if (params.batchSize < 1) {
        newErrors.batchSize = 'Batch size must be at least 1';
      }
      if (params.learningRate < 0) {
        newErrors.learningRate = 'Learning rate must be non-negative';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  async function handleSubmit() {
    if (!validate()) return;

    setSubmitting(true);
    try {
      const finalParams = isManagedExternally
        ? { ...params, trainingImages: trainingImagesPath! }
        : params;

      const result = trainingSchema.safeParse(finalParams);
      if (!result.success) {
        const zodErrors: Record<string, string> = {};
        (result.error.issues || []).forEach((issue) => {
          const field = issue.path.join('.') || 'unknown';
          zodErrors[field] = issue.message;
        });
        setErrors(zodErrors);
        return;
      }

      onSubmit(result.data);
    } finally {
      setSubmitting(false);
    }
  }

  // Single-mode renderers
  function renderNumberInput(label: string, key: keyof typeof params, min: number, step = 1) {
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
          {label}
        </label>
        <input
          id={key}
          type="number"
          min={min}
          step={step}
          value={params[key] as number}
          onChange={(e) => updateParam(key, parseFloat(e.target.value) || 0)}
          className={`w-full px-3 py-2 border rounded-md text-sm bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 ${
            errors[key] ? 'border-red-500' : 'border-slate-300 dark:border-slate-600'
          } focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500`}
        />
        {errors[key] && <p className="text-red-500 dark:text-red-400 text-xs mt-1">{errors[key]}</p>}
      </div>
    );
  }

  function renderSelect(label: string, key: keyof typeof params, options: string[]) {
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
          {label}
        </label>
        <select
          id={key}
          value={params[key] as string}
          onChange={(e) => updateParam(key, e.target.value as any)}
          className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md text-sm bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500"
        >
          {options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  }

  function renderCheckbox(label: string, key: keyof typeof params) {
    return (
      <div key={key} className="flex items-center">
        <input
          id={key}
          type="checkbox"
          checked={params[key] as boolean}
          onChange={(e) => updateParam(key, e.target.checked)}
          className="h-4 w-4 text-slate-900 dark:text-slate-100 border-slate-300 dark:border-slate-600 rounded focus:ring-slate-400 dark:focus:ring-slate-500"
        />
        <label htmlFor={key} className="ml-2 text-sm text-slate-700 dark:text-slate-300">
          {label}
        </label>
      </div>
    );
  }

  function renderTextInput(label: string, key: keyof typeof params, placeholder = '') {
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
          {label}
        </label>
        <input
          id={key}
          type="text"
          value={params[key] as string}
          onChange={(e) => updateParam(key, e.target.value)}
          placeholder={placeholder}
          className={`w-full px-3 py-2 border rounded-md text-sm bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 ${
            errors[key] ? 'border-red-500' : 'border-slate-300 dark:border-slate-600'
          } focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500`}
        />
        {errors[key] && <p className="text-red-500 dark:text-red-400 text-xs mt-1">{errors[key]}</p>}
      </div>
    );
  }

  // Matrix-mode multi-select renderer
  function renderMultiSelect(label: string, key: string, presets: string[]) {
    return (
      <div key={key}>
        <MultiSelectDropdown
          label={label}
          value={matrixValues[key] || []}
          presets={presets}
          onChange={(values) => updateMatrixParam(key, values)}
        />
        {errors[key] && <p className="text-red-500 dark:text-red-400 text-xs mt-1">{errors[key]}</p>}
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-6">
        Anima Training Parameters
      </h2>

      <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }} className="space-y-6">
        {/* Network Parameters */}
        <section>
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
            Network
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {matrixMode ? (
              <>
                {renderMultiSelect('Network Dim', 'networkDim', DEFAULT_NETWORK_DIMS)}
                {renderMultiSelect('Network Alpha', 'networkAlpha', DEFAULT_NETWORK_ALPHAS)}
              </>
            ) : (
              <>
                {renderNumberInput('Network Dim', 'networkDim', 1)}
                {renderNumberInput('Network Alpha', 'networkAlpha', 0)}
              </>
            )}
          </div>
        </section>

        {/* Training Parameters */}
        <section>
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
            Training
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {matrixMode ? (
              <>
                {renderMultiSelect('Learning Rate', 'learningRate', DEFAULT_LEARNING_RATES)}
                {renderMultiSelect('Batch Size', 'batchSize', DEFAULT_BATCH_SIZES)}
                {renderMultiSelect('Epochs', 'epochs', DEFAULT_EPOCHS)}
              </>
            ) : (
              <>
                {renderNumberInput('Learning Rate', 'learningRate', 0, 0.0001)}
                {renderNumberInput('Batch Size', 'batchSize', 1)}
                {renderNumberInput('Epochs', 'epochs', 1)}
              </>
            )}
          </div>
        </section>

        {/* Optimizer & Scheduler */}
        <section>
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
            Optimizer & Scheduler
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {matrixMode ? (
              <>
                {renderMultiSelect('Optimizer', 'optimizer', OPTIMIZERS)}
                {renderMultiSelect('Scheduler', 'scheduler', SCHEDULERS)}
              </>
            ) : (
              <>
                {renderSelect('Optimizer', 'optimizer', OPTIMIZERS)}
                {renderSelect('Scheduler', 'scheduler', SCHEDULERS)}
              </>
            )}
          </div>
        </section>

        {/* Data */}
        <section>
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
            Data
          </h3>
          <div className="grid grid-cols-1 gap-4 max-w-sm">
            {renderTextInput('LoRA Name', 'loraName', 'my-lora')}
          </div>
        </section>

        {/* Precision & Sampling */}
        <section>
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
            Precision & Sampling
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {matrixMode ? (
              <>
                {renderMultiSelect('Mixed Precision', 'mixedPrecision', MIXED_PRECISIONS)}
                {renderMultiSelect('Timestep Sampling', 'timestepSampling', TIMESTEP_SAMPLINGS)}
              </>
            ) : (
              <>
                {renderSelect('Mixed Precision', 'mixedPrecision', MIXED_PRECISIONS)}
                {renderSelect('Timestep Sampling', 'timestepSampling', TIMESTEP_SAMPLINGS)}
              </>
            )}
          </div>
        </section>

        {/* Optimizations */}
        <section>
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
            Optimizations
          </h3>
          <div className="space-y-2">
            {renderCheckbox('Gradient Checkpointing', 'gradientCheckpointing')}
            {renderCheckbox('Cache Latents', 'cacheLatents')}
            {renderCheckbox('Cache Text Encoder', 'cacheTextEncoder')}
          </div>
        </section>

        {/* Submit */}
        <div className="pt-4">
          <button
            type="submit"
            disabled={submitting}
            className="px-6 py-2 bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white rounded-md hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? 'Starting...' : 'Start Training'}
          </button>
        </div>
      </form>
    </div>
  );
}
