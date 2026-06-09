import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

// Mock fetch
const mockFetch = vi.fn();
beforeEach(() => {
  vi.stubGlobal('fetch', mockFetch);
});
afterEach(() => {
  vi.unstubAllGlobals();
});

async function importDashboard() {
  vi.resetModules();
  // Re-mock after resetModules
  vi.stubGlobal('fetch', mockFetch);
  const mod = await import('../Dashboard');
  return mod.Dashboard;
}

function mockSetupReady() {
  mockFetch.mockResolvedValueOnce({
    json: async () => ({ venvReady: true, sdScriptsReady: true, setup: { status: 'success' } }),
  });
}

function mockConfigEmpty() {
  mockFetch.mockResolvedValueOnce({
    json: async () => ({ config: { trainingImagesDir: '', outputDir: '' } }),
  });
}

describe('Dashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders navigation sections', async () => {
    mockSetupReady();
    mockConfigEmpty();

    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('Setup')).toBeInTheDocument();
      expect(screen.getByText('Models')).toBeInTheDocument();
      expect(screen.getByText('Train')).toBeInTheDocument();
      expect(screen.getByText('Jobs')).toBeInTheDocument();
    });
  });

  it('shows Train section by default when setup is complete', async () => {
    mockSetupReady();
    mockConfigEmpty();

    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      const trainButton = screen.getByText('Train').closest('button');
      expect(trainButton).toHaveClass('bg-slate-200');
    });
  });

  it('redirects to Setup when setup is incomplete', async () => {
    // Dashboard GET /api/setup + GET /api/config, then SetupWizard GET /api/setup (initial + poll)
    const incompleteResponse = {
      venvReady: false,
      sdScriptsReady: false,
      setup: { status: 'idle', currentStep: null, steps: {}, updatedAt: new Date().toISOString() },
    };
    mockFetch.mockResolvedValue({
      json: async () => incompleteResponse,
    });
    // Also need config response
    mockFetch.mockResolvedValueOnce({
      json: async () => ({ config: { trainingImagesDir: '', outputDir: '' } }),
    });

    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      // Should show Setup section content, not Train
      const nav = screen.getByRole('navigation');
      const setupButton = nav.querySelector('button[aria-current]') ?? nav.children[0] as HTMLElement;
      // Simpler: check that Setup nav button is active
      const setupButtons = screen.getAllByText('Setup');
      expect(setupButtons.length).toBeGreaterThan(0);
      // Also verify SetupWizard content is visible
      expect(screen.getByText(/setup required/i)).toBeInTheDocument();
    });
  });

  it('greys out non-Setup nav items when setup is incomplete', async () => {
    const incompleteResponse = {
      venvReady: false,
      sdScriptsReady: false,
      setup: { status: 'idle', currentStep: null, steps: {}, updatedAt: new Date().toISOString() },
    };
    mockFetch.mockResolvedValue({
      json: async () => incompleteResponse,
    });
    mockFetch.mockResolvedValueOnce({
      json: async () => ({ config: { trainingImagesDir: '', outputDir: '' } }),
    });

    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      // Train button should be locked (has lock icon or is cursor-not-allowed)
      const trainButton = screen.getByText('Train').closest('button');
      expect(trainButton).toHaveAttribute('disabled');
    });
  });

  it('shows model type tabs in Train section when setup is complete', async () => {
    mockSetupReady();
    mockConfigEmpty();

    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('Anima')).toBeInTheDocument();
      expect(screen.getByText('FLUX')).toBeInTheDocument();
    });
  });

  it('shows Matrix Mode toggle in Train section when setup is complete', async () => {
    mockSetupReady();
    mockConfigEmpty();

    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/Single Run/i)).toBeInTheDocument();
      expect(screen.getByText(/Matrix Run/i)).toBeInTheDocument();
    });
  });

  it('navigates to Setup section on click when setup is complete', async () => {
    mockSetupReady();
    mockConfigEmpty();
    // Subsequent fetch for SetupWizard's own GET
    mockFetch.mockResolvedValue({
      json: async () => ({
        venvReady: true,
        sdScriptsReady: true,
        setup: { status: 'success', currentStep: null, steps: {}, gpu: 'RTX 4090', series: 'ada', cuda: 'cu128', updatedAt: new Date().toISOString() },
      }),
    });

    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/Single Run/i)).toBeInTheDocument();
    });

    const setupBtn = screen.getByText('Setup');
    fireEvent.click(setupBtn);

    await waitFor(() => {
      const setupContent = screen.getByText(/environment is ready/i)
        || screen.getByText(/setup required/i)
        || screen.getByText(/python venv/i);
      expect(setupContent).toBeInTheDocument();
    });
  });

  it('navigates to Jobs section on click when setup is complete', async () => {
    mockSetupReady();
    mockConfigEmpty();

    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/Single Run/i)).toBeInTheDocument();
    });

    const jobsBtn = screen.getByText('Jobs');
    fireEvent.click(jobsBtn);

    await waitFor(() => {
      const trainButton = screen.getByText('Train').closest('button');
      expect(trainButton).not.toHaveClass('bg-slate-200');
    });
  });
});
