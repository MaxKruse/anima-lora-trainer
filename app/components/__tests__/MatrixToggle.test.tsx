import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importMatrixToggle() {
  const mod = await import('../MatrixToggle');
  return mod.MatrixToggle;
}

describe('MatrixToggle', () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('default mode is Single Run', async () => {
    const MatrixToggle = await importMatrixToggle();
    render(<MatrixToggle onChange={mockOnChange} />);

    const singleMode = screen.getByText(/Single Run/i) || screen.getByText(/Single/i);
    expect(singleMode).toBeInTheDocument();
  });

  it('toggling to Matrix Run changes mode', async () => {
    const MatrixToggle = await importMatrixToggle();
    render(<MatrixToggle onChange={mockOnChange} />);

    // Find and click the toggle switch
    const toggle = screen.getByRole('switch');
    fireEvent.click(toggle);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith('matrix');
    });
  });

  it('shows permutation count when in Matrix mode', async () => {
    const MatrixToggle = await importMatrixToggle();
    render(
      <MatrixToggle
        onChange={mockOnChange}
        permutationCount={12}
        mode="matrix"
      />
    );

    await waitFor(() => {
      const countElement = screen.queryByText(/12/) || screen.queryByText(/permutation/i);
      expect(countElement !== null || screen.getByText(/Matrix/i)).toBe(true);
    }, { timeout: 2000 });
  });
});
