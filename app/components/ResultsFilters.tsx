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
    <div className="results-filters">
      {/* Sort control */}
      <div className="filter-group">
        <label htmlFor="sort-select">Sort by</label>
        <select
          id="sort-select"
          role="combobox"
          aria-label="Sort by parameter"
          onChange={(e) => handleSortChange(e.target.value)}
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
        <div key={paramName} className="filter-group">
          <label htmlFor={`filter-${paramName}`}>{paramName}</label>
          <select
            id={`filter-${paramName}`}
            role="combobox"
            aria-label={`Filter by ${paramName}`}
            onChange={(e) => handleFilterChange(paramName, e.target.value)}
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
      <button onClick={handleClear} className="clear-filters-btn">
        Clear filters
      </button>
    </div>
  );
}
