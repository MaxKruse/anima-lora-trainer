import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';

async function importLoraDownload() {
  const mod = await import('../LoraDownload');
  return mod.LoraDownload;
}

describe('LoraDownload', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders download link pointing to correct file path', async () => {
    const LoraDownload = await importLoraDownload();
    render(
      <LoraDownload
        runId="run-001"
        loraFile="my_lora-000001.safetensors"
        exists={true}
      />
    );

    await waitFor(() => {
      const link = screen.getByRole('link');
      expect(link).toHaveAttribute(
        'href',
        '/api/download?runId=run-001&file=my_lora-000001.safetensors'
      );
      expect(link).toBeEnabled();
    });
  });

  it('link disabled when file does not exist', async () => {
    const LoraDownload = await importLoraDownload();
    render(
      <LoraDownload
        runId="run-001"
        loraFile="missing.safetensors"
        exists={false}
      />
    );

    await waitFor(() => {
      const link = screen.getByRole('link');
      expect(link).toHaveAttribute('aria-disabled', 'true');
    });
  });
});
