'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

interface DirectoryPickerProps {
  label: string;
  value: string;
  onChange: (path: string) => void;
  placeholder?: string;
  hint?: string;
  error?: string;
  id?: string;
  autoVerify?: boolean;
}

/**
 * DirectoryPicker — combines a native folder picker (File System Access API)
 * with a manual text input fallback. Persists the selected path.
 *
 * On browsers that support showDirectoryPicker, the "Browse" button opens
 * a native folder selection dialog. The path text input always works as a
 * fallback for manual entry.
 *
 * When autoVerify is true, the directory is checked automatically when the
 * value changes (debounced). When false, the user must click "Check" manually.
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
  const [isPickerSupported, setIsPickerSupported] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [verifyStatus, setVerifyStatus] = useState<'ok' | 'fail' | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Detect File System Access API support
  useEffect(() => {
    setIsPickerSupported(
      typeof window !== 'undefined' && 'showDirectoryPicker' in window
    );
  }, []);

  // Auto-verify when value changes (debounced)
  useEffect(() => {
    if (!autoVerify || !value.trim() || verifying) return;

    // Clear any pending debounce
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(async () => {
      setVerifyStatus(null);
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
      }
    }, 400);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [value, autoVerify, verifying]);

  const handleBrowse = useCallback(async () => {
    if (!isPickerSupported) return;

    try {
      // @ts-ignore — showDirectoryPicker isn't in TS lib yet
      const handle = await window.showDirectoryPicker({
        mode: 'read',
        startIn: 'documents',
      });
      // FileSystemDirectoryHandle.name gives just the folder name.
      // We can't get the full path from the browser for security reasons.
      // So we set a descriptive label and let the user confirm/edit.
      const displayName = handle.name;
      onChange(displayName);
      setVerifyStatus(null);
    } catch {
      // User cancelled — do nothing
    }
  }, [isPickerSupported, onChange]);

  const handleVerify = useCallback(async () => {
    if (!value.trim()) return;
    setVerifying(true);
    setVerifyStatus(null);

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
  }, [value]);

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

      <div className="flex gap-2">
        <input
          ref={inputRef}
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

        {isPickerSupported && (
          <button
            type="button"
            onClick={handleBrowse}
            className="px-3 py-2 text-sm bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-600 rounded-md hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors whitespace-nowrap"
            title="Open folder picker"
          >
            Browse
          </button>
        )}

        {!autoVerify && (
          <button
            type="button"
            onClick={handleVerify}
            disabled={verifying || !value.trim()}
            className="px-3 py-2 text-sm bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-600 rounded-md hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
            title="Check if directory exists"
          >
            {verifying ? '...' : 'Check'}
          </button>
        )}

        {autoVerify && verifying && (
          <span className="px-2 py-1 text-sm text-slate-400 dark:text-slate-500">
            ...
          </span>
        )}

        {autoVerify && !verifying && verifyStatus === null && value.trim() && (
          <span className="px-2 py-1 text-sm text-slate-400 dark:text-slate-500">
            ...
          </span>
        )}

        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="px-2 py-2 text-sm text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
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
