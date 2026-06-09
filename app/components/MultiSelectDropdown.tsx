'use client';

import { useState, useRef, useEffect, useMemo } from 'react';

interface MultiSelectDropdownProps {
  label: string;
  value: string[];
  presets: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
}

export function MultiSelectDropdown({
  label,
  value,
  presets,
  onChange,
  placeholder = 'Type to add...',
}: MultiSelectDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setInputValue('');
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const visiblePresets = useMemo(() => {
    if (!inputValue.trim()) return presets;
    const lower = inputValue.toLowerCase();
    return presets.filter((p) => p.toLowerCase().includes(lower));
  }, [inputValue, presets]);

  const hasCustomOption = inputValue.trim() !== '' && !presets.includes(inputValue.trim());

  function handleAdd(valueToAdd: string) {
    const trimmed = valueToAdd.trim();
    if (!trimmed || value.includes(trimmed)) return;
    onChange([...value, trimmed]);
    setInputValue('');
  }

  function handleRemove(index: number) {
    const newValue = [...value];
    newValue.splice(index, 1);
    onChange(newValue);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleAdd(inputValue);
    } else if (event.key === 'Backspace' && inputValue === '' && value.length > 0) {
      handleRemove(value.length - 1);
    }
  }

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
        {label}
      </label>

      {/* Selected tags */}
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2">
          {value.map((val, idx) => (
            <span
              key={val}
              className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-md"
            >
              <span>{val}</span>
              <button
                type="button"
                onClick={() => handleRemove(idx)}
                className="hover:text-red-500 dark:hover:text-red-400"
                aria-label={`Remove ${val}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Input field */}
      <input
        ref={inputRef}
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => setIsOpen(true)}
        placeholder={placeholder}
        className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md text-sm bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500"
      />

      {/* Dropdown */}
      {isOpen && (visiblePresets.length > 0 || hasCustomOption) && (
        <div className="absolute z-20 w-full mt-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md shadow-lg max-h-48 overflow-y-auto">
          {/* Custom option */}
          {hasCustomOption && (
            <button
              type="button"
              onMouseDown={(e) => {
                e.preventDefault();
                handleAdd(inputValue);
                setIsOpen(false);
              }}
              className="w-full text-left px-3 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 first:rounded-t-md"
            >
              Add &quot;{inputValue.trim()}&quot;
            </button>
          )}

          {/* Preset options */}
          {visiblePresets.map((preset) => (
            <button
              key={preset}
              type="button"
              onMouseDown={(e) => {
                e.preventDefault();
                handleAdd(preset);
              }}
              className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-700 last:rounded-b-md ${
                value.includes(preset)
                  ? 'text-slate-400 dark:text-slate-500 line-through'
                  : 'text-slate-700 dark:text-slate-300'
              }`}
            >
              {preset}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
