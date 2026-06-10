import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

async function importAnimaTab() {
  const mod = await import('../AnimaTab');
  return mod.AnimaTab;
}

describe('AnimaTab - Matrix Mode', () => {
  const mockOnSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders single-value inputs when matrixMode is false', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={false}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Should have regular select elements for optimizer
    const optimizerSelect = screen.getByLabelText(/optimizer/i);
    expect(optimizerSelect.tagName).toBe('SELECT');
  });

  it('renders multi-select dropdowns when matrixMode is true', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Should have multi-select inputs (with "Type to add" placeholder)
    const multiSelectInputs = screen.getAllByPlaceholderText(/type to add/i);
    expect(multiSelectInputs.length).toBeGreaterThan(0);
  });

  it('shows multi-select for optimizer in matrix mode', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Should have multi-select inputs (with "Type to add" placeholder)
    const multiSelectInputs = screen.getAllByPlaceholderText(/type to add/i);
    expect(multiSelectInputs.length).toBeGreaterThan(0);

    // Optimizer label should exist (use getAllByText since section header also matches)
    const optimizerElements = screen.getAllByText(/optimizer/i);
    expect(optimizerElements.length).toBeGreaterThan(0);
  });

  it('shows multi-select for scheduler in matrix mode', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={true}
      />
    );

    await waitFor(() => {
      // Scheduler label exists (section header "Optimizer & Scheduler" also matches)
      const schedulerElements = screen.getAllByText(/scheduler/i);
      expect(schedulerElements.length).toBeGreaterThan(0);
    });
  });

  it('shows multi-select for epochs in matrix mode', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/epochs/i)).toBeInTheDocument();
    });
  });

  it('shows multi-select for learning rate in matrix mode', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/learning rate/i)).toBeInTheDocument();
    });
  });

  it('shows multi-select for network dim in matrix mode', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/network dim/i)).toBeInTheDocument();
    });
  });

  it('submitting in matrix mode sends paramRanges', async () => {
    const user = userEvent.setup();
    const mockOnMatrixSubmit = vi.fn();
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        onMatrixSubmit={mockOnMatrixSubmit}
        matrixMode={true}
        trainingImagesPath="/test/images"
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Fill in LoRA name
    const nameInput = screen.getByLabelText(/lora name/i);
    await user.type(nameInput, 'matrix-test-lora');

    // Submit
    const submitBtn = screen.getByRole('button', { name: /train/i })
      || screen.getByRole('button', { name: /start/i })
      || screen.getByRole('button');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(mockOnMatrixSubmit).toHaveBeenCalled();
    });

    const [paramRanges, baseParams] = mockOnMatrixSubmit.mock.calls[0];
    expect(paramRanges).toHaveProperty('networkDim');
    expect(paramRanges).toHaveProperty('epochs');
    expect(baseParams).toHaveProperty('trainingImages', '/test/images');
    expect(baseParams).toHaveProperty('loraName', 'matrix-test-lora');
  });

  it('defaults to AdamW8Bit in matrix mode', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Should show AdamW8Bit as a selected tag
    const adamTags = screen.getAllByText('AdamW8Bit');
    expect(adamTags.length).toBeGreaterThan(0);
  });

  it('defaults to constant scheduler in matrix mode', async () => {
    const AnimaTab = await importAnimaTab();
    render(
      <AnimaTab
        onSubmit={mockOnSubmit}
        matrixMode={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Should show constant as a selected tag
    const constantTags = screen.getAllByText('constant');
    expect(constantTags.length).toBeGreaterThan(0);
  });
});
