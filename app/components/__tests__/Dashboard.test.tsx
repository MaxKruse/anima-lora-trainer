import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importDashboard() {
  const mod = await import('../Dashboard');
  return mod.Dashboard;
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
    const Dashboard = await importDashboard();
    render(<Dashboard />);

    expect(screen.getByText('Setup')).toBeInTheDocument();
    expect(screen.getByText('Models')).toBeInTheDocument();
    expect(screen.getByText('Train')).toBeInTheDocument();
    expect(screen.getByText('Jobs')).toBeInTheDocument();
  });

  it('shows Train section by default', async () => {
    const Dashboard = await importDashboard();
    render(<Dashboard />);

    // Train section should be active by default
    await waitFor(() => {
      const trainButton = screen.getByText('Train').closest('button');
      expect(trainButton).toHaveClass('bg-slate-200');
    });
  });

  it('shows model type tabs in Train section', async () => {
    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('Anima')).toBeInTheDocument();
      expect(screen.getByText('FLUX')).toBeInTheDocument();
    });
  });

  it('shows Matrix Mode toggle in Train section', async () => {
    const Dashboard = await importDashboard();
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/Single Run/i)).toBeInTheDocument();
      expect(screen.getByText(/Matrix Run/i)).toBeInTheDocument();
    });
  });

  it('navigates to Setup section on click', async () => {
    const Dashboard = await importDashboard();
    render(<Dashboard />);

    const setupBtn = screen.getByText('Setup');
    fireEvent.click(setupBtn);

    await waitFor(() => {
      expect(screen.getByText(/GPU/i) || screen.getByText(/Setup/i)).toBeInTheDocument();
    });
  });

  it('navigates to Jobs section on click', async () => {
    const Dashboard = await importDashboard();
    render(<Dashboard />);

    const jobsBtn = screen.getByText('Jobs');
    fireEvent.click(jobsBtn);

    await waitFor(() => {
      const trainButton = screen.getByText('Train').closest('button');
      expect(trainButton).not.toHaveClass('bg-slate-200');
    });
  });
});
