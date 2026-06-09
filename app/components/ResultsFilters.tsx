'use client';

import { useState, useMemo, useCallback } from 'react';
import { ResultEntry } from '../lib/results-loader';

interface ResultsFiltersProps {
  results: ResultEntry[];
  onFilterChange?: (filters: Record<string, string>) => void;
  onSortChange?: (paramName: string) => void;
}

/**
 * UI controls for filtering and sorting results by parameter values.
 */
export function ResultsFilters({
  results,
  onFilterChange,
  onSortChange,
}: ResultsFiltersProps) {
  // Collect all unique parameter names and their values
  const paramOptions = useMemo(() => {
    const params: Record<string, Set<string>> = {};

    for (const result of results) {
      for (const [key, value] of Object.entries(result.params)) {
        if (!params[key]) params[key] = new Set();
        params[key].add(String(value));
      }
    }

    // Convert sets to sorted arrays
    return Object.fromEntries(
      Object.entries(params).map(([k, v]) => [k, [...v].sort()])
    );
  }, [results]);

  const paramNames = Object.keys(paramOptions).sort();

  const handleFilterChange = useCallback(
    (paramName: string, value: string) => {
      if (!onFilterChange) return;

      const newFilters: Record<string, string> = {};
      if (value) {
        newFilters[paramName] = value;
      }
      onFilterChange(newFilters);
    },
    [onFilterChange]
  );

  const handleSortChange = useCallback(
    (paramName: string) => {
      if (paramName && onSortChange) {
        onSortChange(paramName);
      }
    },
    [onSortChange]
  );

  const handleClear = useCallback(() => {
    onFilterChange?.({});
  }, [onFilterChange]);

  return (
    <div className="flex flex-wrap items-end gap-3 p-4 border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800">
      {/* Sort control */}
      <div className="flex flex-col gap-1">
        <label htmlFor="sort-select" className="text-xs font-medium text-slate-500 dark:text-slate-400">
          Sort by
        </label>
        <select
          id="sort-select"
          role="combobox"
          aria-label="Sort by parameter"
          onChange={(e) => handleSortChange(e.target.value)}
          className="px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500"
        >
          <option value="">None</option>
          {paramNames.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      {/* Filter dropdowns for each parameter */}
      {paramNames.map((paramName) => (
        <div key={paramName} className="flex flex-col gap-1">
          <label htmlFor={`filter-${paramName}`} className="text-xs font-medium text-slate-500 dark:text-slate-400">
            {paramName}
          </label>
          <select
            id={`filter-${paramName}`}
            role="combobox"
            aria-label={`Filter by ${paramName}`}
            onChange={(e) => handleFilterChange(paramName, e.target.value)}
            className="px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500"
          >
            <option value="">All</option>
            {paramOptions[paramName].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
      ))}

      {/* Clear button */}
      <button
        onClick={handleClear}
        className="px-3 py-1.5 text-sm text-slate-600 dark:text-slate-300 border border-slate-300 dark:border-slate-600 rounded-md hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
      >
        Clear filters
      </button>
    </div>
  );
}
