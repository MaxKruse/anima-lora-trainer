import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importComparisonView() {
  const mod = await import('../ComparisonView');
  return mod.ComparisonView;
}

const sampleResults = [
  { params: { 'network-dim': 32, 'learning-rate': 0.0001 }, loraFile: 'lora-000001.safetensors', imageFile: 'eval_a.png', status: 'completed', inferenceTimeMs: 1500 },
  { params: { 'network-dim': 64, 'learning-rate': 0.001 }, loraFile: 'lora-000002.safetensors', imageFile: 'eval_b.png', status: 'completed', inferenceTimeMs: 1600 },
  { params: { 'network-dim': 128, 'learning-rate': 0.0001 }, loraFile: 'lora-000003.safetensors', imageFile: 'eval_c.png', status: 'completed', inferenceTimeMs: 1700 },
];

describe('ComparisonView', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders selected results in a horizontal row', async () => {
    const ComparisonView = await importComparisonView();
    render(
      <ComparisonView
        results={sampleResults}
        selectedIndices={[0, 1]}
      />
    );

    await waitFor(() => {
      const panels = screen.getAllByRole('region');
      expect(panels).toHaveLength(2);
    });
  });

  it('each panel shows image, params, and LoRA file link', async () => {
    const ComparisonView = await importComparisonView();
    render(
      <ComparisonView
        results={sampleResults}
        selectedIndices={[0, 1]}
      />
    );

    await waitFor(() => {
      // Each panel should show param values (in separate spans)
      expect(screen.getByText('32')).toBeInTheDocument();
      expect(screen.getByText('64')).toBeInTheDocument();

      // Each panel should show the lora file name
      expect(screen.getByText('lora-000001.safetensors')).toBeInTheDocument();
      expect(screen.getByText('lora-000002.safetensors')).toBeInTheDocument();
    });
  });

  it('minimum 2 selections required', async () => {
    const ComparisonView = await importComparisonView();
    render(
      <ComparisonView
        results={sampleResults}
        selectedIndices={[0]}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/select at least 2/i)).toBeInTheDocument();
    });
  });

  it('deselect removes from comparison', async () => {
    const onDeselect = vi.fn();

    const ComparisonView = await importComparisonView();
    render(
      <ComparisonView
        results={sampleResults}
        selectedIndices={[0, 1]}
        onDeselect={onDeselect}
      />
    );

    await waitFor(() => {
      const removeButtons = screen.getAllByRole('button', { name: /remove/i });
      fireEvent.click(removeButtons[0]);
    });

    expect(onDeselect).toHaveBeenCalledWith(0);
  });
});
