import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Mock fetch for API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

async function importSetupWizard() {
  const mod = await import('../SetupWizard');
  return mod.SetupWizard;
}

describe('SetupWizard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders "Detect GPU" button initially', async () => {
    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    expect(screen.getByRole('button', { name: /detect\s*gpu/i })).toBeInTheDocument();
  });

  it('displays GPU name and CUDA version after successful detection', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        gpu: 'NVIDIA GeForce RTX 4090',
        series: 'ada',
        cuda: 'cu128',
        status: 'ok',
      }),
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    const button = screen.getByRole('button', { name: /detect\s*gpu/i });
    fireEvent.click(button);

    await waitFor(() => {
      expect(screen.getByText(/RTX 4090/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/cu128/i)).toBeInTheDocument();
  });

  it('shows error message when detection fails', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: 'nvidia-smi not found', status: 'error' }),
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    const button = screen.getByRole('button', { name: /detect\s*gpu/i });
    fireEvent.click(button);

    await waitFor(() => {
      expect(screen.getByText(/nvidia-smi not found/i)).toBeInTheDocument();
    });
  });

  it('shows "Environment ready" when pyproject.toml exists', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        gpu: 'NVIDIA GeForce RTX 4090',
        series: 'ada',
        cuda: 'cu128',
        pyprojectPath: '/project/pyproject.toml',
        status: 'ok',
      }),
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    const button = screen.getByRole('button', { name: /detect\s*gpu/i });
    fireEvent.click(button);

    await waitFor(() => {
      expect(screen.getByText(/environment ready/i)).toBeInTheDocument();
    });
  });
});
