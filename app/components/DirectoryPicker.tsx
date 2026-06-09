'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

interface DirectoryPickerProps {
  label: string;
  value: string;
  onChange: (path: string) => void;
  placeholder?: string;
  hint?: string;
  error?: string;
  id?: string;
  /**
   * When true (default), auto-verify the directory exists on the server.
   * Set to false for directories that are created automatically (e.g., output dir).
   */
  autoVerify?: boolean;
}

/**
 * DirectoryPicker — text input with optional auto-verification.
 *
 * When autoVerify is true, the directory path is checked against the server
 * automatically when the value changes (debounced). When false, no verification
 * occurs (useful for directories that will be created on the fly).
 */
export function DirectoryPicker({
  label,
  value,
  onChange,
  placeholder = '/path/to/directory',
  hint,
  error,
  id,
  autoVerify = true,
}: DirectoryPickerProps) {
  const [verifying, setVerifying] = useState(false);
  const [verifyStatus, setVerifyStatus] = useState<'ok' | 'fail' | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevValueRef = useRef<string | undefined>(undefined);

  // Auto-verify when value changes (debounced)
  useEffect(() => {
    if (!autoVerify || !value.trim()) return;
    // Skip if value hasn't actually changed (prevents flicker on re-renders)
    if (value === prevValueRef.current) return;
    prevValueRef.current = value;

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(async () => {
      if (verifying) return; // Guard against overlapping calls
      setVerifying(true);
      try {
        const res = await fetch('/api/config/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: value }),
        });
        const data = await res.json();
        setVerifyStatus(res.ok && data.exists ? 'ok' : 'fail');
      } catch {
        setVerifyStatus('fail');
      } finally {
        setVerifying(false);
      }
    }, 400);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [value, autoVerify]);

  const handleClear = useCallback(() => {
    onChange('');
    setVerifyStatus(null);
  }, [onChange]);

  return (
    <div className="space-y-2">
      <label
        htmlFor={id}
        className="block text-sm font-medium text-slate-700 dark:text-slate-300"
      >
        {label}
      </label>

      {/* Input row */}
      <div className="flex gap-2 items-center">
        <input
          id={id}
          type="text"
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            setVerifyStatus(null);
          }}
          placeholder={placeholder}
          className={`flex-1 px-3 py-2 border rounded-md text-sm font-mono bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 ${
            error
              ? 'border-red-500 focus:ring-red-500'
              : verifyStatus === 'ok'
                ? 'border-green-500 focus:ring-green-500'
                : verifyStatus === 'fail'
                  ? 'border-yellow-500 focus:ring-yellow-500'
                  : 'border-slate-300 dark:border-slate-600 focus:ring-slate-400 dark:focus:ring-slate-500'
          } focus:outline-none focus:ring-2`}
        />

        {verifying && (
          <span className="px-2 py-1 text-sm text-slate-400 dark:text-slate-500">
            ...
          </span>
        )}

        {verifyStatus === null && value.trim() && !verifying && (
          <span className="px-2 py-1 text-sm text-slate-400 dark:text-slate-500">
            ...
          </span>
        )}

        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="px-2 py-2 text-sm text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors shrink-0"
            title="Clear"
          >
            ✕
          </button>
        )}
      </div>

      {/* Status indicators */}
      {verifyStatus === 'ok' && (
        <p className="text-xs text-green-600 dark:text-green-400 flex items-center gap-1">
          <span>✓</span> Directory exists
        </p>
      )}
      {verifyStatus === 'fail' && (
        <p className="text-xs text-yellow-600 dark:text-yellow-400 flex items-center gap-1">
          <span>!</span> Directory not found or inaccessible
        </p>
      )}

      {error && (
        <p className="text-xs text-red-500 dark:text-red-400">{error}</p>
      )}

      {hint && !error && verifyStatus !== 'fail' && (
        <p className="text-xs text-slate-400 dark:text-slate-500">{hint}</p>
      )}
    </div>
  );
}
