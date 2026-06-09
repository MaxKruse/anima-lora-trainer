import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importResultsGrid() {
  const mod = await import('../ResultsGrid');
  return mod.ResultsGrid;
}

const sampleResults = [
  {
    params: { 'network-dim': 32, 'learning-rate': 0.0001 },
    loraFile: 'lora-000001.safetensors',
    imageFile: 'eval_perm-a.png',
    status: 'completed',
    inferenceTimeMs: 1500,
  },
  {
    params: { 'network-dim': 64, 'learning-rate': 0.001 },
    loraFile: 'lora-000002.safetensors',
    imageFile: 'eval_perm-b.png',
    status: 'completed',
    inferenceTimeMs: 1600,
  },
  {
    params: { 'network-dim': 128, 'learning-rate': 0.0001 },
    loraFile: 'lora-000003.safetensors',
    imageFile: null,
    status: 'failed',
    inferenceTimeMs: null,
  },
];

describe('ResultsGrid', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders one card per permutation result', async () => {
    const ResultsGrid = await importResultsGrid();
    render(<ResultsGrid results={sampleResults} />);

    await waitFor(() => {
      // Should render 3 cards
      const cards = screen.getAllByRole('article');
      expect(cards).toHaveLength(3);
    });
  });

  it('each card shows parameter values and evaluation image', async () => {
    const ResultsGrid = await importResultsGrid();
    render(<ResultsGrid results={sampleResults} />);

    await waitFor(() => {
      // Parameter values should be visible
      expect(screen.getByText('network-dim: 32')).toBeInTheDocument();
      expect(screen.getByText('network-dim: 64')).toBeInTheDocument();
    });

    // Images should be rendered for completed results
    await waitFor(() => {
      const images = screen.getAllByRole('img');
      expect(images.length).toBeGreaterThanOrEqual(2);
    });
  });

  it('missing images show placeholder', async () => {
    const ResultsGrid = await importResultsGrid();
    render(<ResultsGrid results={sampleResults} />);

    await waitFor(() => {
      // The third result has no image and failed status
      expect(screen.getByText(/no image/i)).toBeInTheDocument();
    });
  });

  it('clicking card selects it for comparison', async () => {
    const onSelectChange = vi.fn();

    const ResultsGrid = await importResultsGrid();
    render(
      <ResultsGrid
        results={sampleResults}
        selectedIds={[]}
        onSelectChange={onSelectChange}
      />
    );

    await waitFor(() => {
      const cards = screen.getAllByRole('article');
      fireEvent.click(cards[0]);
    });

    expect(onSelectChange).toHaveBeenCalled();
  });
});
