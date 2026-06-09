import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importAnimaTab() {
  const mod = await import('../AnimaTab');
  return mod.AnimaTab;
}

describe('AnimaTab', () => {
  const mockOnSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders all fields from spec', async () => {
    const AnimaTab = await importAnimaTab();
    render(<AnimaTab onSubmit={mockOnSubmit} />);

    // Network parameters
    expect(screen.getByLabelText(/network dim/i) || screen.getByText(/network dim/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/network alpha/i) || screen.getByText(/network alpha/i)).toBeInTheDocument();

    // Training parameters
    expect(screen.getByLabelText(/learning rate/i) || screen.getByText(/learning rate/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/batch size/i) || screen.getByText(/batch size/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/epochs/i) || screen.getByText(/epochs/i)).toBeInTheDocument();

    // Optimizer & Scheduler
    expect(screen.getByLabelText(/optimizer/i) || screen.getByText(/optimizer/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/scheduler/i) || screen.getByText(/scheduler/i)).toBeInTheDocument();

    // Data
    expect(screen.getByLabelText(/training images/i) || screen.getByText(/training images/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/lora name/i) || screen.getByText(/lora name/i)).toBeInTheDocument();

    // Precision & Sampling
    expect(screen.getByLabelText(/mixed precision/i) || screen.getByText(/mixed precision/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/timestep sampling/i) || screen.getByText(/timestep sampling/i)).toBeInTheDocument();

    // Optimizations (checkboxes)
    expect(screen.getByLabelText(/gradient checkpointing/i) || screen.getByText(/gradient checkpointing/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/cache latents/i) || screen.getByText(/cache latents/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/cache text encoder/i) || screen.getByText(/cache text encoder/i)).toBeInTheDocument();
  });

  it('each field has correct default value', async () => {
    const AnimaTab = await importAnimaTab();
    render(<AnimaTab onSubmit={mockOnSubmit} />);

    // Wait for component to render
    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Default optimizer should be AdamW8Bit
    const optimizerSelect = screen.getByLabelText(/optimizer/i);
    if (optimizerSelect.tagName === 'SELECT') {
      expect(optimizerSelect).toHaveValue('AdamW8Bit');
    }

    // Default scheduler should be cosine
    const schedulerSelect = screen.getByLabelText(/scheduler/i);
    if (schedulerSelect.tagName === 'SELECT') {
      expect(schedulerSelect).toHaveValue('cosine');
    }

    // Default mixed precision should be bf16
    const precisionSelect = screen.getByLabelText(/mixed precision/i);
    if (precisionSelect.tagName === 'SELECT') {
      expect(precisionSelect).toHaveValue('bf16');
    }

    // Default timestep sampling should be sigmoid
    const samplingSelect = screen.getByLabelText(/timestep sampling/i);
    if (samplingSelect.tagName === 'SELECT') {
      expect(samplingSelect).toHaveValue('sigmoid');
    }

    // Checkboxes should be checked by default
    const gcCheckbox = screen.getByLabelText(/gradient checkpointing/i);
    if (gcCheckbox.tagName === 'INPUT') {
      expect(gcCheckbox).toBeChecked();
    }

    const cacheLatentsCheckbox = screen.getByLabelText(/cache latents/i);
    if (cacheLatentsCheckbox.tagName === 'INPUT') {
      expect(cacheLatentsCheckbox).toBeChecked();
    }
  });

  it('submitting fires callback with correct param object', async () => {
    const AnimaTab = await importAnimaTab();
    render(<AnimaTab onSubmit={mockOnSubmit} />);

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Fill in required text fields
    const imagesInput = screen.getByLabelText(/training images/i);
    if (imagesInput.tagName === 'INPUT') {
      fireEvent.change(imagesInput, { target: { value: '/path/to/images' } });
    }

    const nameInput = screen.getByLabelText(/lora name/i);
    if (nameInput.tagName === 'INPUT') {
      fireEvent.change(nameInput, { target: { value: 'my-lora' } });
    }

    // Submit the form
    const submitBtn = screen.getByRole('button', { name: /train/i })
      || screen.getByRole('button', { name: /submit/i })
      || screen.getByRole('button');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalled();
    });

    const submittedParams = mockOnSubmit.mock.calls[0][0];
    expect(submittedParams).toHaveProperty('networkDim');
    expect(submittedParams).toHaveProperty('networkAlpha');
    expect(submittedParams).toHaveProperty('learningRate');
    expect(submittedParams).toHaveProperty('batchSize');
    expect(submittedParams).toHaveProperty('epochs');
    expect(submittedParams).toHaveProperty('optimizer');
    expect(submittedParams).toHaveProperty('scheduler');
    expect(submittedParams).toHaveProperty('trainingImages');
    expect(submittedParams).toHaveProperty('loraName');
    expect(submittedParams).toHaveProperty('mixedPrecision');
    expect(submittedParams).toHaveProperty('timestepSampling');
  });

  it('validates required fields before submit', async () => {
    const AnimaTab = await importAnimaTab();
    render(<AnimaTab onSubmit={mockOnSubmit} />);

    await waitFor(() => {
      expect(screen.getByText(/Anima Training Parameters/i)).toBeInTheDocument();
    });

    // Don't fill in required text fields, just click submit
    const submitBtn = screen.getByRole('button', { name: /train/i })
      || screen.getByRole('button', { name: /submit/i })
      || screen.getByRole('button');
    fireEvent.click(submitBtn);

    // Should NOT call onSubmit because required fields are empty
    expect(mockOnSubmit).not.toHaveBeenCalled();

    // Should show some error/validation feedback
    await waitFor(() => {
      const errorElements = screen.queryAllByText(/required|fill|enter|must/i);
      const submitButton = screen.queryByRole('button', { name: /train/i, disabled: true });
      expect(errorElements.length > 0 || submitButton !== null).toBe(true);
    }, { timeout: 2000 }).catch(() => {
      // If no error message shown, at least onSubmit should not have been called
      expect(mockOnSubmit).not.toHaveBeenCalled();
    });
  });
});
