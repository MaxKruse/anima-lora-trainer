import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

async function importEvaluateButton() {
  const mod = await import('../EvaluateButton');
  return mod.EvaluateButton;
}

describe('EvaluateButton', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders when run has completed status', async () => {
    const EvaluateButton = await importEvaluateButton();
    render(<EvaluateButton runId="run-001" runStatus="completed" />);

    await waitFor(() => {
      const button = screen.getByRole('button', { name: /evaluate/i });
      expect(button).toBeInTheDocument();
      expect(button).toBeEnabled();
    });
  });

  it('disabled during evaluation', async () => {
    const EvaluateButton = await importEvaluateButton();
    render(<EvaluateButton runId="run-001" runStatus="evaluating" />);

    await waitFor(() => {
      const button = screen.getByRole('button', { name: /evaluat/i });
      expect(button).toBeDisabled();
    });
  });

  it('shows progress during evaluation', async () => {
    const EvaluateButton = await importEvaluateButton();
    render(<EvaluateButton runId="run-001" runStatus="evaluating" />);

    await waitFor(() => {
      const elements = screen.getAllByText(/evaluating/i);
      expect(elements.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('triggers results refresh after evaluation completes', async () => {
    const onResultsRefresh = vi.fn();

    // POST /api/evaluate succeeds
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ runId: 'run-001', status: 'started' }),
    });

    // Simulated polling: GET /api/evaluate returns results
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        prompt: 'cat dog bird',
        total: 3,
        completed: 3,
        failed: 0,
        results: [
          { perm_name: 'perm-a', status: 'completed', inference_time_ms: 1500 },
        ],
      }),
    });

    const EvaluateButton = await importEvaluateButton();
    render(
      <EvaluateButton
        runId="run-001"
        runStatus="completed"
        onResultsRefresh={onResultsRefresh}
      />
    );

    const button = screen.getByRole('button', { name: /evaluate/i });
    fireEvent.click(button);

    // After evaluation completes, onResultsRefresh should be called
    await waitFor(() => {
      expect(onResultsRefresh).toHaveBeenCalled();
    }, { timeout: 5000 });
  });
});
