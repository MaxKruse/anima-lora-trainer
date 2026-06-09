import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';

async function importAnimaTab() {
  const mod = await import('../AnimaTab');
  return mod.AnimaTab;
}

describe('AnimaTab - Data section', () => {
  const mockOnSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  afterEach(() => {
    cleanup();
  });

  it('does not show training images display when path is managed externally', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        trainingImagesPath="/some/path"
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Should NOT show the redundant "Training Images" read-only display
    const trainingImagesLabels = screen.queryAllByText(/training images/i);
    // The label should not appear because we removed the redundant display
    expect(trainingImagesLabels.length).toBe(0);
  });

  it('does not show "Set in directory picker above" hint', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        trainingImagesPath="/some/path"
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    expect(screen.queryByText(/set in directory picker/i)).not.toBeInTheDocument();
  });

  it('still shows LoRA Name field when path is managed externally', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        trainingImagesPath="/some/path"
      />
    );

    await waitFor(() => {
      expect(screen.getByLabelText(/lora name/i)).toBeInTheDocument();
    });
  });

  it('shows Data section header', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        trainingImagesPath="/some/path"
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Data')).toBeInTheDocument();
    });
  });
});
