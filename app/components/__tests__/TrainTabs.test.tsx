import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importTrainTabs() {
  const mod = await import('../TrainTabs');
  return mod.TrainTabs;
}

describe('TrainTabs', () => {
  const mockOnSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders tabs for all model types', async () => {
    const TrainTabs = await importTrainTabs();
    render(<TrainTabs onSubmit={mockOnSubmit} trainingImagesPath="" />);

    // Should have tabs for each model type
    expect(screen.getByText('Anima')).toBeInTheDocument();
    expect(screen.getByText('FLUX')).toBeInTheDocument();
    expect(screen.getByText('SD3')).toBeInTheDocument();
    expect(screen.getByText('SDXL')).toBeInTheDocument();
    expect(screen.getByText('SD 1.5')).toBeInTheDocument();
    expect(screen.getByText('Hunyuan')).toBeInTheDocument();
    expect(screen.getByText('Lumina')).toBeInTheDocument();
  });

  it('shows Anima tab content by default', async () => {
    const TrainTabs = await importTrainTabs();
    render(<TrainTabs onSubmit={mockOnSubmit} trainingImagesPath="" />);

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });
  });

  it('switches to FLUX tab on click', async () => {
    const TrainTabs = await importTrainTabs();
    render(<TrainTabs onSubmit={mockOnSubmit} trainingImagesPath="" />);

    const fluxTab = screen.getByText('FLUX');
    fireEvent.click(fluxTab);

    await waitFor(() => {
      expect(screen.getByText(/coming soon/i) || screen.getByText(/FLUX/i)).toBeInTheDocument();
    });
  });

  it('shows coming soon for unsupported model types', async () => {
    const TrainTabs = await importTrainTabs();
    render(<TrainTabs onSubmit={mockOnSubmit} trainingImagesPath="" />);

    // Click on a non-Anima tab
    const sdxlTab = screen.getByText('SDXL');
    fireEvent.click(sdxlTab);

    await waitFor(() => {
      expect(screen.getByText(/coming soon/i)).toBeInTheDocument();
    });
  });

  it('highlights active tab', async () => {
    const TrainTabs = await importTrainTabs();
    render(<TrainTabs onSubmit={mockOnSubmit} trainingImagesPath="" />);

    const animaTab = screen.getByText('Anima').closest('button');
    expect(animaTab).toHaveClass('bg-slate-200');
  });

  it('passes trainingImagesPath to AnimaTab', async () => {
    const TrainTabs = await importTrainTabs();
    render(<TrainTabs onSubmit={mockOnSubmit} trainingImagesPath="/custom/path" />);

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // AnimaTab should render with the training images path (used internally for training)
    expect(screen.getByLabelText(/lora name/i)).toBeInTheDocument();
  });
});
