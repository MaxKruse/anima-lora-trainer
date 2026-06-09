'use client';

import { useState } from 'react';

interface MatrixToggleProps {
  onChange?: (mode: 'single' | 'matrix', permutationCount?: number) => void;
  permutationCount?: number;
  mode?: 'single' | 'matrix';
}

export function MatrixToggle({ onChange, permutationCount, mode: initialMode = 'single' }: MatrixToggleProps) {
  const [mode, setMode] = useState<'single' | 'matrix'>(initialMode);

  function handleToggle() {
    const newMode = mode === 'single' ? 'matrix' : 'single';
    setMode(newMode);
    onChange?.(newMode, permutationCount);
  }

  return (
    <div className="flex items-center gap-4 mb-6">
      <span className={`text-sm font-medium ${mode === 'single' ? 'text-blue-600' : 'text-gray-500'}`}>
        Single Run
      </span>

      <button
        onClick={handleToggle}
        className={`relative w-12 h-6 rounded-full transition-colors ${
          mode === 'matrix' ? 'bg-blue-600' : 'bg-gray-300'
        }`}
        role="switch"
        aria-checked={mode === 'matrix'}
        aria-label="Toggle between Single Run and Matrix Run modes"
      >
        <span
          className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
            mode === 'matrix' ? 'translate-x-6' : 'translate-x-0'
          }`}
        />
      </button>

      <span className={`text-sm font-medium ${mode === 'matrix' ? 'text-blue-600' : 'text-gray-500'}`}>
        Matrix Run
      </span>

      {mode === 'matrix' && permutationCount !== undefined && permutationCount > 0 && (
        <span className="text-sm text-gray-500 ml-2">
          ({permutationCount} permutations)
        </span>
      )}
    </div>
  );
}
