'use client';

import { useState } from 'react';
import { trainingSchema, type TrainingParams } from '../lib/training-schema';

interface AnimaTabProps {
  onSubmit: (params: TrainingParams) => void;
  /**
   * When provided, the training images path is pre-filled and the field
   * is hidden (managed by the parent directory picker).
   */
  trainingImagesPath?: string;
}

const OPTIMIZERS = ['AdamW8Bit', 'AdamW', 'Prodigy', 'Lion', 'Adafactor'];
const SCHEDULERS = ['constant', 'cosine', 'linear', 'constant_with_warmup', 'cosine_with_restarts'];
const MIXED_PRECISIONS = ['fp16', 'bf16', 'no'];
const TIMESTEP_SAMPLINGS = ['sigma', 'uniform', 'sigmoid', 'shift', 'flux_shift'];

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

export function AnimaTab({ onSubmit, trainingImagesPath }: AnimaTabProps) {
  const isManagedExternally = trainingImagesPath !== undefined;
  const [params, setParams] = useState({
    ...DEFAULT_PARAMS,
    trainingImages: trainingImagesPath || '',
    loraName: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  function updateParam<K extends keyof typeof params>(key: K, value: (typeof params)[K]) {
    setParams((prev) => ({ ...prev, [key]: value }));
    // Clear error for this field
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

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  async function handleSubmit() {
    if (!validate()) return;

    setSubmitting(true);
    try {
      // Use external path if provided
      const finalParams = isManagedExternally
        ? { ...params, trainingImages: trainingImagesPath! }
        : params;

      // Validate against zod schema
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

  function renderNumberInput(label: string, key: keyof typeof params, min: number, step = 1) {
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-gray-700 mb-1">
          {label}
        </label>
        <input
          id={key}
          type="number"
          min={min}
          step={step}
          value={params[key] as number}
          onChange={(e) => updateParam(key, parseFloat(e.target.value) || 0)}
          className={`w-full px-3 py-2 border rounded-md text-sm ${
            errors[key] ? 'border-red-500' : 'border-gray-300'
          }`}
        />
        {errors[key] && <p className="text-red-500 text-xs mt-1">{errors[key]}</p>}
      </div>
    );
  }

  function renderSelect(label: string, key: keyof typeof params, options: string[]) {
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-gray-700 mb-1">
          {label}
        </label>
        <select
          id={key}
          value={params[key] as string}
          onChange={(e) => updateParam(key, e.target.value as any)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
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
          className="h-4 w-4 text-blue-600 border-gray-300 rounded"
        />
        <label htmlFor={key} className="ml-2 text-sm text-gray-700">
          {label}
        </label>
      </div>
    );
  }

  function renderTextInput(label: string, key: keyof typeof params, placeholder = '') {
    return (
      <div key={key}>
        <label htmlFor={key} className="block text-sm font-medium text-gray-700 mb-1">
          {label}
        </label>
        <input
          id={key}
          type="text"
          value={params[key] as string}
          onChange={(e) => updateParam(key, e.target.value)}
          placeholder={placeholder}
          className={`w-full px-3 py-2 border rounded-md text-sm ${
            errors[key] ? 'border-red-500' : 'border-gray-300'
          }`}
        />
        {errors[key] && <p className="text-red-500 text-xs mt-1">{errors[key]}</p>}
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-xl font-bold mb-6">Anima Training Parameters</h2>

      <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }} className="space-y-6">
        {/* Network Parameters */}
        <section>
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Network
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {renderNumberInput('Network Dim', 'networkDim', 1)}
            {renderNumberInput('Network Alpha', 'networkAlpha', 0)}
          </div>
        </section>

        {/* Training Parameters */}
        <section>
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Training
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {renderNumberInput('Learning Rate', 'learningRate', 0, 0.0001)}
            {renderNumberInput('Batch Size', 'batchSize', 1)}
            {renderNumberInput('Epochs', 'epochs', 1)}
          </div>
        </section>

        {/* Optimizer & Scheduler */}
        <section>
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Optimizer & Scheduler
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {renderSelect('Optimizer', 'optimizer', OPTIMIZERS)}
            {renderSelect('Scheduler', 'scheduler', SCHEDULERS)}
          </div>
        </section>

        {/* Data */}
        <section>
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Data
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {isManagedExternally ? (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Training Images
                </label>
                <div className="px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md text-sm font-mono text-gray-500 dark:text-gray-400 truncate">
                  {trainingImagesPath}
                </div>
                <p className="text-xs text-gray-400 mt-1">Set in directory picker above</p>
              </div>
            ) : (
              renderTextInput('Training Images Path', 'trainingImages', '/path/to/images')
            )}
            {renderTextInput('LoRA Name', 'loraName', 'my-lora')}
          </div>
        </section>

        {/* Precision & Sampling */}
        <section>
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Precision & Sampling
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {renderSelect('Mixed Precision', 'mixedPrecision', MIXED_PRECISIONS)}
            {renderSelect('Timestep Sampling', 'timestepSampling', TIMESTEP_SAMPLINGS)}
          </div>
        </section>

        {/* Optimizations */}
        <section>
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
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
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting ? 'Starting...' : 'Start Training'}
          </button>
        </div>
      </form>
    </div>
  );
}
