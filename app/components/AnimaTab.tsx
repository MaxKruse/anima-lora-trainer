'use client';

import { useState, useEffect, useMemo } from 'react';
import { trainingSchema, type TrainingParams } from '../lib/training-schema';
import { MultiSelectDropdown } from './MultiSelectDropdown';

interface AnimaTabProps {
  onSubmit: (params: TrainingParams) => void;
  /**
   * Called in matrix mode with paramRanges and baseParams.
   * If not provided, the component POSTs to /api/train/matrix directly.
   */
  onMatrixSubmit?: (paramRanges: Record<string, string>, baseParams: Record<string, any>) => void;
  /**
   * Called when the permutation count changes in matrix mode.
   */
  onPermutationCountChange?: (count: number) => void;
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
const DEFAULT_RESOLUTIONS = ['768', '1024'];

const DEFAULT_PARAMS: Omit<TrainingParams, 'trainingImages' | 'loraName'> = {
  networkDim: 8,
  networkAlpha: 8,
  learningRate: 1e-4,
  batchSize: 4,
  epochs: 10,
  resolution: 1024,
  optimizer: 'AdamW8Bit',
  scheduler: 'constant',
  mixedPrecision: 'bf16',
  timestepSampling: 'sigmoid',
  gradientCheckpointing: true,
  cacheLatents: true,
  cacheTextEncoder: false,
  captionTagDropoutRate: 0.05,
  keepTokens: 1,
};

// Default matrix values (single value arrays for initial state)
const DEFAULT_MATRIX_VALUES: Record<string, string[]> = {
  networkDim: ['8'],
  networkAlpha: ['8'],
  learningRate: ['0.0001'],
  batchSize: ['4'],
  epochs: ['10'],
  resolution: ['1024'],
  optimizer: ['AdamW8Bit'],
  scheduler: ['constant'],
  mixedPrecision: ['bf16'],
  timestepSampling: ['sigmoid'],
};

export function AnimaTab({ onSubmit, onMatrixSubmit, onPermutationCountChange, trainingImagesPath, matrixMode = false }: AnimaTabProps) {
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
  const [nameAvailable, setNameAvailable] = useState<boolean | null>(null);
  const [nameChecking, setNameChecking] = useState(false);

  // Calculate permutation count for matrix mode
  const permutationCount = useMemo(() => {
    if (!matrixMode) return 0;
    return Object.values(matrixValues).reduce((total, values) => {
      return values.length > 0 ? total * values.length : 0;
    }, 1);
  }, [matrixMode, matrixValues]);

  // Report permutation count to parent
  useEffect(() => {
    onPermutationCountChange?.(permutationCount);
  }, [permutationCount, onPermutationCountChange]);

  // Show required-field errors immediately on mount
  useEffect(() => {
    validate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Check name availability when loraName changes
  useEffect(() => {
    if (!params.loraName || params.loraName.length < 2) {
      setNameAvailable(null);
      return;
    }

    setNameChecking(true);
    let cancelled = false;
    fetch(`/api/check-name?name=${encodeURIComponent(params.loraName)}`)
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled) {
          setNameAvailable(data.available);
          setNameChecking(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setNameAvailable(null);
          setNameChecking(false);
        }
      });
    return () => { cancelled = true; };
  }, [params.loraName]);

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

    // Block submit if name is taken
    if (nameAvailable === false) {
      setErrors((prev) => ({ ...prev, loraName: 'Name already exists — choose a different one' }));
      return;
    }

    // Matrix mode: build matrix payload and submit directly
    if (matrixMode) {
      // Block excessive permutations
      const MAX_PERMS = 500;
      if (permutationCount > MAX_PERMS) {
        setErrors(prev => ({ ...prev, _submit: `Too many permutations (${permutationCount}). Maximum is ${MAX_PERMS}.` }));
        return;
      }

      setSubmitting(true);
      try {
        // Build paramRanges from matrixValues
        const paramRanges: Record<string, string> = {};
        for (const [key, values] of Object.entries(matrixValues)) {
          if (values.length > 0) {
            paramRanges[key] = values.join(',');
          }
        }

        // Build baseParams from non-matrix fields
        const baseParams = {
          trainingImages: isManagedExternally ? trainingImagesPath : params.trainingImages,
          loraName: params.loraName,
          maxSteps: (params as Record<string, any>).maxSteps,
          mixedPrecision: params.mixedPrecision,
          timestepSampling: params.timestepSampling,
          gradientCheckpointing: params.gradientCheckpointing,
          cacheLatents: params.cacheLatents,
          cacheTextEncoder: params.cacheTextEncoder,
          captionTagDropoutRate: params.captionTagDropoutRate,
          keepTokens: (params as Record<string, any>).keepTokens,
        };

        if (onMatrixSubmit) {
          onMatrixSubmit(paramRanges, baseParams);
        } else {
          const res = await fetch('/api/train/matrix', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paramRanges, baseParams }),
          });

          const data = await res.json();

          if (!res.ok) {
            setErrors(prev => ({ ...prev, _submit: data.error || 'Matrix training failed' }));
          }
        }
      } catch (err: any) {
        setErrors(prev => ({ ...prev, _submit: err.message || 'Network error' }));
      } finally {
        setSubmitting(false);
      }
      return;
    }

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
  function renderNumberInput(label: string, key: keyof typeof params, min: number, step = 1, max?: number) {
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
          {label}
        </label>
        <input
          id={key}
          type="number"
          min={min}
          max={max}
          step={step}
          value={params[key] as number}
          onChange={(e) => updateParam(key, parseFloat(e.target.value.replace(',', '.')) || 0)}
          className={`w-full px-3 py-2 border rounded-md text-sm bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 ${
            errors[key] ? 'border-red-500' : 'border-slate-300 dark:border-slate-600'
          } focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500`}
        />
        {errors[key] && <p className="text-red-500 dark:text-red-400 text-xs mt-1">{errors[key]}</p>}
      </div>
    );
  }

  function renderOptionalNumberInput(label: string, key: string, min: number, placeholder = '') {
    const value = (params as Record<string, any>)[key];
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
          {label}
        </label>
        <input
          id={key}
          type="number"
          min={min}
          value={value ?? ''}
          placeholder={placeholder}
          onChange={(e) => {
            const raw = e.target.value.replace(',', '.');
            (params as Record<string, any>)[key] = raw === '' ? undefined : parseInt(raw, 10);
            setParams({ ...params });
            if (errors[key]) {
              setErrors((prev) => {
                const next = { ...prev };
                delete next[key];
                return next;
              });
            }
          }}
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

  function renderTextInput(label: string, key: keyof typeof params, placeholder = '', required = false) {
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
          {label}{required && <span className="text-red-500 ml-1">*</span>}
        </label>
        <input
          id={key}
          type="text"
          value={params[key] as string}
          onChange={(e) => updateParam(key, e.target.value)}
          placeholder={placeholder}
          className={`w-full px-3 py-2 border rounded-md text-sm bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 ${
            errors[key] ? 'border-red-500 focus:ring-red-500' : 'border-slate-300 dark:border-slate-600'
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
    <div>
      <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-6">
        Anima Training Parameters
      </h2>

      <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
        {/* Top row: Network + Training + Optimizer/Scheduler side by side */}
        <div className="grid grid-cols-3 gap-8">
          {/* Network Parameters */}
          <section>
            <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
              Network
            </h3>
            <div className="space-y-3">
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
            <div className="space-y-3">
              {matrixMode ? (
                <>
                  {renderMultiSelect('Learning Rate', 'learningRate', DEFAULT_LEARNING_RATES)}
                  {renderMultiSelect('Batch Size', 'batchSize', DEFAULT_BATCH_SIZES)}
                  {renderMultiSelect('Epochs', 'epochs', DEFAULT_EPOCHS)}
                  {renderOptionalNumberInput('Max Steps (optional)', 'maxSteps', 1, 'If set, overrides epochs')}
                  {renderOptionalNumberInput('Repeats per image (optional)', 'repeats', 1, 'Auto-calculated if empty')}
                </>
              ) : (
                <>
                  {renderNumberInput('Learning Rate', 'learningRate', 0, 0.0001)}
                  {renderNumberInput('Batch Size', 'batchSize', 1)}
                  {renderNumberInput('Epochs', 'epochs', 1)}
                  {renderOptionalNumberInput('Max Steps (optional)', 'maxSteps', 1, 'If set, overrides epochs')}
                  {renderOptionalNumberInput('Repeats per image (optional)', 'repeats', 1, 'Auto-calculated if empty')}
                </>
              )}
            </div>
          </section>

          {/* Optimizer & Scheduler */}
          <section>
            <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
              Optimizer & Scheduler
            </h3>
            <div className="space-y-3">
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
        </div>

        {/* Bottom row: Data + Precision/Sampling + Optimizations side by side */}
        <div className="grid grid-cols-3 gap-8 mt-8">
          {/* Data */}
          <section>
            <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
              Data
            </h3>
            <div className="space-y-3">
              <div>
                {renderTextInput('LoRA Name', 'loraName', 'my-lora', true)}
                {!nameChecking && params.loraName.length >= 2 && nameAvailable !== null && (
                  <p className={`text-xs mt-1 ${nameAvailable ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {nameAvailable ? '✓ Name is available' : '✗ Name already exists — choose a different one'}
                  </p>
                )}
                {nameChecking && (
                  <p className="text-xs mt-1 text-slate-400 dark:text-slate-500">Checking...</p>
                )}
              </div>
              {matrixMode
                ? renderMultiSelect('Resolution', 'resolution', DEFAULT_RESOLUTIONS)
                : renderNumberInput('Resolution', 'resolution', 768, 16, 1024)}
            </div>
          </section>

          {/* Precision & Sampling */}
          <section>
            <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
              Precision & Sampling
            </h3>
            <div className="space-y-3">
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
              Caption
            </h3>
            <div className="space-y-2">
              <div>
                <label htmlFor="keepTokens" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Keep Tokens
                </label>
                <input
                  id="keepTokens"
                  type="number"
                  min={0}
                  max={10}
                  step={1}
                  value={(params as Record<string, any>).keepTokens ?? 1}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    (params as Record<string, any>).keepTokens = isNaN(val) ? 1 : Math.min(10, Math.max(0, val));
                    setParams({ ...params });
                  }}
                  className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md text-sm bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500"
                />
              </div>
              <div>
                <label htmlFor="captionTagDropoutRate" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Caption Tag Dropout Rate
                </label>
                <input
                  id="captionTagDropoutRate"
                  type="text"
                  inputMode="decimal"
                  value={params.captionTagDropoutRate}
                  onChange={(e) => {
                    const raw = e.target.value.replace(',', '.');
                    const val = parseFloat(raw);
                    updateParam('captionTagDropoutRate', isNaN(val) ? 0 : Math.min(1, Math.max(0, val)));
                  }}
                  className={`w-full px-3 py-2 border rounded-md text-sm bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 ${
                    errors.captionTagDropoutRate ? 'border-red-500' : 'border-slate-300 dark:border-slate-600'
                  } focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500`}
                />
                {errors.captionTagDropoutRate && <p className="text-red-500 dark:text-red-400 text-xs mt-1">{errors.captionTagDropoutRate}</p>}
              </div>
            </div>
          </section>

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
        </div>

        {/* Permutation count warning (matrix mode) */}
        {matrixMode && permutationCount > 0 && (
          <div className={`mt-6 p-3 rounded-lg ${
            permutationCount > 100
              ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
              : permutationCount > 20
                ? 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800'
                : 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800'
          }`}>
            <p className="text-sm font-medium">
              {permutationCount} training permutations
            </p>
            {permutationCount > 100 && (
              <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                Warning: This will create {permutationCount} separate training jobs.
                Estimated time: {Math.round(permutationCount * 15 / 60)} hours.
              </p>
            )}
          </div>
        )}

        {/* Submit */}
        <div className="pt-6">
          {errors._submit && (
            <p className="text-red-600 dark:text-red-400 text-sm mb-3">{errors._submit}</p>
          )}
          <button
            type="submit"
            disabled={submitting}
            className="w-full px-6 py-2 bg-slate-900 dark:bg-slate-100 dark:text-slate-900 text-white rounded-md hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {submitting
              ? matrixMode ? 'Starting Matrix Training...' : 'Starting...'
              : matrixMode ? `Start Matrix Training (${permutationCount})`
              : 'Start Training'}
          </button>
        </div>
      </form>
    </div>
  );
}
