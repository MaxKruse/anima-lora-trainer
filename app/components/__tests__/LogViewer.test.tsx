import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importLogViewer() {
  const mod = await import('../LogViewer');
  return mod.LogViewer;
}

const sampleLines = [
  'INFO: Training started',
  'INFO: Epoch 1/10 completed',
  'WARNING: Low VRAM',
  'ERROR: Failed to allocate tensor',
  'INFO: Retrying...',
];

describe('LogViewer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders log lines in order', async () => {
    const LogViewer = await importLogViewer();
    render(<LogViewer lines={sampleLines} />);

    await waitFor(() => {
      expect(screen.getByText('INFO: Training started')).toBeInTheDocument();
      expect(screen.getByText('INFO: Epoch 1/10 completed')).toBeInTheDocument();
    });

    // Verify order: first line appears before last line in DOM
    const container = screen.getByRole('log');
    const text = container.textContent || '';
    const trainingIdx = text.indexOf('Training started');
    const retryIdx = text.indexOf('Retrying...');
    expect(trainingIdx).toBeLessThan(retryIdx);
  });

  it('search input filters visible lines', async () => {
    const LogViewer = await importLogViewer();
    render(<LogViewer lines={sampleLines} />);

    await waitFor(() => {
      const searchInput = screen.getByRole('searchbox');
      fireEvent.change(searchInput, { target: { value: 'ERROR' } });
    });

    // After filtering, only ERROR lines should be visible
    await waitFor(() => {
      expect(screen.queryByText('INFO: Training started')).not.toBeInTheDocument();
      expect(screen.getByText('ERROR: Failed to allocate tensor')).toBeInTheDocument();
    });
  });

  it('auto-scrolls to latest line', async () => {
    const LogViewer = await importLogViewer();
    render(<LogViewer lines={sampleLines} autoScroll={true} />);

    await waitFor(() => {
      const container = screen.getByRole('log');
      expect(container).toHaveAttribute('data-auto-scroll', 'true');
    });
  });

  it('"Failed" lines highlighted in red', async () => {
    const LogViewer = await importLogViewer();
    render(<LogViewer lines={sampleLines} />);

    await waitFor(() => {
      const errorLine = screen.getByText('ERROR: Failed to allocate tensor');
      expect(errorLine).toHaveClass('text-red-600');
      expect(errorLine).toHaveClass('dark:text-red-400');
    });
  });
});
