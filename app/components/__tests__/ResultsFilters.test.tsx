import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importResultsFilters() {
  const mod = await import('../ResultsFilters');
  return mod.ResultsFilters;
}

const sampleResults = [
  { params: { 'network-dim': 32, 'optimizer': 'AdamW' }, loraFile: 'a.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 100 },
  { params: { 'network-dim': 64, 'optimizer': 'AdamW' }, loraFile: 'b.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 200 },
  { params: { 'network-dim': 32, 'optimizer': 'Prodigy' }, loraFile: 'c.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 300 },
  { params: { 'network-dim': 128, 'optimizer': 'Prodigy' }, loraFile: 'd.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 400 },
];

describe('ResultsFilters', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('shows filter dropdowns for each parameter dimension', async () => {
    const ResultsFilters = await importResultsFilters();
    render(<ResultsFilters results={sampleResults} />);

    await waitFor(() => {
      // Should have filter dropdowns for each unique parameter
      const filterSelects = screen.getAllByRole('combobox');
      // At least 3: 1 sort + 2 filter (network-dim, optimizer)
      expect(filterSelects.length).toBeGreaterThanOrEqual(3);

      // Verify filter labels exist
      expect(screen.getByLabelText('Filter by network-dim')).toBeInTheDocument();
      expect(screen.getByLabelText('Filter by optimizer')).toBeInTheDocument();
    });
  });

  it('filtering narrows visible results', async () => {
    const onFilterChange = vi.fn();

    const ResultsFilters = await importResultsFilters();
    render(
      <ResultsFilters
        results={sampleResults}
        onFilterChange={onFilterChange}
      />
    );

    // Find the network-dim select and change it
    await waitFor(() => {
      const selects = screen.getAllByRole('combobox');
      // Find the one labeled network-dim
      const dimSelect = selects.find(
        (s) => s.getAttribute('aria-label') === 'Filter by network-dim'
      );
      if (dimSelect) {
        fireEvent.change(dimSelect, { target: { value: '32' } });
      }
    });

    expect(onFilterChange).toHaveBeenCalled();
  });

  it('sorting reorders results by selected parameter', async () => {
    const onSortChange = vi.fn();

    const ResultsFilters = await importResultsFilters();
    render(
      <ResultsFilters
        results={sampleResults}
        onSortChange={onSortChange}
      />
    );

    await waitFor(() => {
      const sortSelect = screen.getByRole('combobox', { name: /sort/i });
      fireEvent.change(sortSelect, { target: { value: 'network-dim' } });
    });

    expect(onSortChange).toHaveBeenCalledWith('network-dim');
  });

  it('clearing filters shows all results', async () => {
    const onFilterChange = vi.fn();

    const ResultsFilters = await importResultsFilters();
    render(
      <ResultsFilters
        results={sampleResults}
        onFilterChange={onFilterChange}
      />
    );

    // Find and click the clear button
    await waitFor(() => {
      const clearBtn = screen.getByRole('button', { name: /clear/i });
      fireEvent.click(clearBtn);
    });

    // Should call onFilterChange with empty filters
    expect(onFilterChange).toHaveBeenCalledWith({});
  });
});
