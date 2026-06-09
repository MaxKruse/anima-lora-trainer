import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// Mock fetch for API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock setInterval/setTimeout for polling
const mockSetInterval = vi.spyOn(global, 'setInterval').mockImplementation((fn: any, ms: number) => {
  (fn as any)(); // Run immediately for tests
  return 1 as any;
});
const mockClearInterval = vi.spyOn(global, 'clearInterval').mockImplementation(() => {});

async function importSetupWizard() {
  vi.resetModules();
  const mod = await import('../SetupWizard');
  return mod.SetupWizard;
}

const defaultGetResponse = {
  venvReady: false,
  sdScriptsReady: false,
  setup: {
    status: 'idle',
    currentStep: null,
    steps: {},
    updatedAt: new Date().toISOString(),
  },
};

const readyGetResponse = {
  venvReady: true,
  sdScriptsReady: true,
  setup: {
    status: 'success',
    currentStep: null,
    steps: {
      'detect-gpu': { status: 'done', output: 'RTX 4090 (ada), CUDA 12.8' },
      'generate-pyproject': { status: 'done', output: 'Wrote pyproject.toml' },
      'clean-venv': { status: 'done', output: 'Removed old .venv and uv.lock' },
      'uv-sync': { status: 'done', output: 'Installed 42 packages' },
      'clone-sd-scripts': { status: 'done', output: 'Cloned to sd-scripts' },
      'done': { status: 'done' },
    },
    gpu: 'NVIDIA GeForce RTX 4090',
    series: 'ada',
    cuda: 'cu128',
    updatedAt: new Date().toISOString(),
  },
};

describe('SetupWizard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    mockSetInterval.mockRestore();
    mockClearInterval.mockRestore();
  });

  it('shows "Setup Environment" when nothing is ready', async () => {
    mockFetch.mockResolvedValue({
      json: async () => defaultGetResponse,
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /setup environment/i })).toBeInTheDocument();
    });
  });

  it('shows "Re-install environment" when everything is ready', async () => {
    mockFetch.mockResolvedValue({
      json: async () => readyGetResponse,
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /re-install environment/i })).toBeInTheDocument();
    });
  });

  it('shows readiness badges for venv and sd-scripts', async () => {
    mockFetch.mockResolvedValue({
      json: async () => ({
        venvReady: true,
        sdScriptsReady: false,
        setup: {
          status: 'idle',
          currentStep: null,
          steps: {},
          updatedAt: new Date().toISOString(),
        },
      }),
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    await waitFor(() => {
      // Use getAllByText since "sd-scripts" appears in both badge and description
      const venvBadges = screen.getAllByText(/python venv/i);
      expect(venvBadges.length).toBeGreaterThan(0);
      expect(venvBadges[0]).toBeInTheDocument();

      const sdBadges = screen.getAllByText(/sd-scripts/i);
      expect(sdBadges.length).toBeGreaterThan(0);
      expect(sdBadges[0]).toBeInTheDocument();
    });
  });

  it('triggers POST and starts polling when setup button is clicked', async () => {
    // Initial GET
    mockFetch.mockResolvedValueOnce({
      json: async () => defaultGetResponse,
    });
    // POST
    mockFetch.mockResolvedValueOnce({
      json: async () => ({ message: 'Setup started', setup: { status: 'running' } }),
    });
    // Subsequent GET polls
    mockFetch.mockResolvedValue({
      json: async () => defaultGetResponse,
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /setup environment/i })).toBeInTheDocument();
    });

    const button = screen.getByRole('button', { name: /setup environment/i });
    act(() => {
      fireEvent.click(button);
    });

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/setup', { method: 'POST' });
    });
  });

  it('displays GPU info after successful setup', async () => {
    mockFetch.mockResolvedValue({
      json: async () => readyGetResponse,
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    await waitFor(() => {
      expect(screen.getByText(/environment is ready/i)).toBeInTheDocument();
    });
    // Use getAllByText since RTX 4090 appears in both GPU info and step output
    const gpuElements = screen.getAllByText(/RTX 4090/i);
    expect(gpuElements.length).toBeGreaterThan(0);
    expect(screen.getByText(/cu128/i)).toBeInTheDocument();
  });

  it('shows step progress when setup is running', async () => {
    const runningResponse = {
      venvReady: false,
      sdScriptsReady: false,
      setup: {
        status: 'running',
        currentStep: 'uv-sync',
        steps: {
          'detect-gpu': { status: 'done', output: 'RTX 4090' },
          'generate-pyproject': { status: 'done', output: 'Wrote pyproject.toml' },
          'clean-venv': { status: 'done', output: 'Removed' },
          'uv-sync': { status: 'running', output: 'Installing dependencies...' },
          'clone-sd-scripts': { status: 'pending' },
          'done': { status: 'pending' },
        },
        updatedAt: new Date().toISOString(),
      },
    };

    mockFetch.mockResolvedValue({
      json: async () => runningResponse,
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    await waitFor(() => {
      expect(screen.getByText(/detect gpu/i)).toBeInTheDocument();
      expect(screen.getByText(/install python dependencies/i)).toBeInTheDocument();
    });
  });

  it('shows error message when setup fails', async () => {
    const errorResponse = {
      venvReady: false,
      sdScriptsReady: false,
      setup: {
        status: 'error',
        currentStep: null,
        steps: {
          'detect-gpu': { status: 'error', output: 'nvidia-smi not found' },
          'generate-pyproject': { status: 'pending' },
          'clean-venv': { status: 'pending' },
          'uv-sync': { status: 'pending' },
          'clone-sd-scripts': { status: 'pending' },
          'done': { status: 'pending' },
        },
        error: 'nvidia-smi not found. Ensure NVIDIA drivers are installed.',
        updatedAt: new Date().toISOString(),
      },
    };

    mockFetch.mockResolvedValue({
      json: async () => errorResponse,
    });

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    await waitFor(() => {
      expect(screen.getByText(/setup failed/i)).toBeInTheDocument();
    });
    // Use getAllByText since the error text appears in both step output and error banner
    const errorElements = screen.getAllByText(/nvidia-smi not found/i);
    expect(errorElements.length).toBeGreaterThan(0);
  });

  it('falls back to setup mode when GET check fails', async () => {
    mockFetch.mockRejectedValueOnce(new Error('network error'));

    const SetupWizard = await importSetupWizard();
    render(<SetupWizard />);

    await waitFor(() => {
      // Should show "Setup required" state with the setup button
      expect(screen.getByRole('button', { name: /setup environment/i })).toBeInTheDocument();
    });
  });
});
