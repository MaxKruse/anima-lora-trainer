import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

async function importModelDownloader() {
  const mod = await import('../ModelDownloader');
  return mod.ModelDownloader;
}

const pendingModel = {
  name: 'diffusion_model',
  status: 'pending' as const,
  progress: 0,
  expectedSizeBytes: 4_180_000_000,
};

const downloadedModel = {
  name: 'diffusion_model',
  status: 'downloaded' as const,
  progress: 100,
  expectedSizeBytes: 4_180_000_000,
};

const downloadingModel = {
  name: 'diffusion_model',
  status: 'downloading' as const,
  progress: 45,
  expectedSizeBytes: 4_180_000_000,
  canAbort: true,
};

describe('ModelDownloader', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders button for each pending model', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [
          { ...pendingModel, name: 'diffusion_model' },
          { ...pendingModel, name: 'vae', expectedSizeBytes: 254_000_000 },
          { ...pendingModel, name: 'text_encoder', expectedSizeBytes: 1_200_000_000 },
        ],
      }),
    });

    const ModelDownloader = await importModelDownloader();
    render(<ModelDownloader />);

    await waitFor(() => {
      expect(screen.getByText(/diffusion_model/i)).toBeInTheDocument();
    });

    // Should have 3 download buttons
    const buttons = screen.getAllByRole('button', { name: /download/i });
    expect(buttons).toHaveLength(3);
  });

  it('shows circular progress indicator during download', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ models: [{ ...downloadingModel, progress: 45 }] }),
    });

    const ModelDownloader = await importModelDownloader();
    render(<ModelDownloader />);

    await waitFor(() => {
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    // Should show percentage text
    expect(screen.getByText('45%')).toBeInTheDocument();
  });

  it('shows abort button on downloading model', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ models: [downloadingModel] }),
    });

    const ModelDownloader = await importModelDownloader();
    render(<ModelDownloader />);

    await waitFor(() => {
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    // Abort button should exist (visible on hover)
    const abortButton = screen.getByRole('button', { name: /abort download/i });
    expect(abortButton).toBeInTheDocument();
  });

  it('calls DELETE on abort button click', async () => {
    // Initial status fetch
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [downloadingModel] }),
    });

    // Abort response
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: 'aborted' }),
    });

    const ModelDownloader = await importModelDownloader();
    render(<ModelDownloader />);

    await waitFor(() => {
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    const abortButton = screen.getByRole('button', { name: /abort download/i });
    fireEvent.click(abortButton);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/models?modelName=diffusion_model'),
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });

  it('shows checkmark when download completes', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ models: [downloadedModel] }),
    });

    const ModelDownloader = await importModelDownloader();
    render(<ModelDownloader />);

    await waitFor(() => {
      const checkmark = screen.queryByText('✓');
      expect(checkmark).toBeInTheDocument();
    });
  });

  it('shows error message when download fails', async () => {
    // Initial status — pending
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [pendingModel] }),
    });

    // Download fails
    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: 'Network error', status: 'failed' }),
    });

    const ModelDownloader = await importModelDownloader();
    render(<ModelDownloader />);

    await waitFor(() => {
      expect(screen.getByText(/diffusion_model/i)).toBeInTheDocument();
    });

    // Click the download button
    const buttons = screen.getAllByRole('button', { name: /download/i });
    fireEvent.click(buttons[0]);

    await waitFor(() => {
      expect(screen.getByText(/Network error/i)).toBeInTheDocument();
    });
  });
});
