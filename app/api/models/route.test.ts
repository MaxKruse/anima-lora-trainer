import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextResponse } from 'next/server';

// --- Mock next/server ---
vi.mock('next/server', () => ({
  NextResponse: {
    json: (data: any, init?: ResponseInit) => ({
      json: async () => data,
      status: init?.status ?? 200,
      headers: new Headers(init?.headers),
    }),
  },
}));

// --- Mock model-manifest ---
vi.mock('../../lib/model-manifest', () => ({
  getResolvedModelManifest: () => Promise.resolve([
    {
      name: 'diffusion_model',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/diffusion_models/anima-base-v1.0.safetensors',
      expectedSizeBytes: 4_182_218_328,
    },
    {
      name: 'vae',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/vae/qwen_image_vae.safetensors',
      expectedSizeBytes: 253_806_246,
    },
    {
      name: 'text_encoder',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/text_encoders/qwen_3_06b_base.safetensors',
      expectedSizeBytes: 1_192_135_096,
    },
  ]),
}));

// --- Mock model-downloader ---
const mockCheckLocal = vi.fn();
const mockDownload = vi.fn();
vi.mock('../../lib/model-downloader', () => ({
  checkLocalModel: (...args: any[]) => mockCheckLocal(...args),
  downloadModel: (...args: any[]) => mockDownload(...args),
}));

// --- Mock fs ---
vi.mock('fs', () => ({
  default: {
    existsSync: vi.fn(() => false),
  },
  existsSync: vi.fn(() => false),
}));

async function importRoute() {
  return await import('./route');
}

describe('/api/models', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Re-import to reset the activeDownloads map
    vi.resetModules();
  });

  it('GET returns status of all models', async () => {
    mockCheckLocal.mockReturnValue({ exists: false });

    const route = await importRoute();
    const response = await route.GET();
    const body = await response.json();

    expect(body.models).toHaveLength(3);
    expect(body.models[0]).toHaveProperty('name');
    expect(body.models[0]).toHaveProperty('status');
    expect(body.models[0]).toHaveProperty('canAbort');
  });

  it('GET shows downloaded status for existing models', async () => {
    mockCheckLocal.mockReturnValue({
      exists: true,
      sizeBytes: 4_180_000_000,
      downloadPercent: 100,
    });

    const route = await importRoute();
    const response = await route.GET();
    const body = await response.json();

    expect(body.models[0].status).toBe('downloaded');
  });

  it('POST triggers download of a specific model and returns immediately', async () => {
    mockCheckLocal.mockReturnValue({ exists: false });
    mockDownload.mockResolvedValue(undefined);

    const route = await importRoute();
    const response = await route.POST(new Request('http://localhost/api/models', {
      method: 'POST',
      body: JSON.stringify({ modelName: 'diffusion_model' }),
    }));
    const body = await response.json();

    expect(body.status).toBe('started');
    // Download runs in background, so it may not have been called yet
    // Wait a tick for the background promise
    await new Promise(resolve => setTimeout(resolve, 10));
    expect(mockDownload).toHaveBeenCalled();
  });

  it('POST returns 409 if model already downloaded', async () => {
    mockCheckLocal.mockReturnValue({
      exists: true,
      sizeBytes: 4_180_000_000,
      downloadPercent: 100,
    });

    const route = await importRoute();
    const response = await route.POST(new Request('http://localhost/api/models', {
      method: 'POST',
      body: JSON.stringify({ modelName: 'diffusion_model' }),
    }));

    expect(response.status).toBe(409);
  });

  it('DELETE aborts an active download', async () => {
    mockCheckLocal.mockReturnValue({ exists: false });
    // Make download never resolve so it stays active
    mockDownload.mockReturnValue(new Promise(() => {}));

    const route = await importRoute();

    // Start a download
    await route.POST(new Request('http://localhost/api/models', {
      method: 'POST',
      body: JSON.stringify({ modelName: 'diffusion_model' }),
    }));

    // Verify it shows as downloading with canAbort
    const getStatus = await route.GET();
    const statusBody = await getStatus.json();
    expect(statusBody.models[0].canAbort).toBe(true);

    // Abort the download
    const deleteResponse = await route.DELETE(
      new Request('http://localhost/api/models?modelName=diffusion_model', {
        method: 'DELETE',
      })
    );
    const deleteBody = await deleteResponse.json();

    expect(deleteBody.status).toBe('aborted');
  });

  it('DELETE returns 404 if no active download', async () => {
    mockCheckLocal.mockReturnValue({ exists: false });

    const route = await importRoute();
    const response = await route.DELETE(
      new Request('http://localhost/api/models?modelName=diffusion_model', {
        method: 'DELETE',
      })
    );

    expect(response.status).toBe(404);
  });

  it('DELETE returns 400 if modelName missing', async () => {
    const route = await importRoute();
    const response = await route.DELETE(
      new Request('http://localhost/api/models', {
        method: 'DELETE',
      })
    );

    expect(response.status).toBe(400);
  });
});
