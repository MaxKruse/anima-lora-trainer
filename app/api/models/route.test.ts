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
  getModelManifest: () => [
    {
      name: 'diffusion_model',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/diffusion_models/anima-base-v1.0.safetensors',
      localPath: 'models/anima/diffusion_models/anima-base-v1.0.safetensors',
      expectedSizeBytes: 4_180_000_000,
    },
    {
      name: 'vae',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/vae/qwen_image_vae.safetensors',
      localPath: 'models/anima/vae/qwen_image_vae.safetensors',
      expectedSizeBytes: 254_000_000,
    },
    {
      name: 'text_encoder',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/text_encoders/qwen_3_06b_base.safetensors',
      localPath: 'models/anima/text_encoders/qwen_3_06b_base.safetensors',
      expectedSizeBytes: 1_200_000_000,
    },
  ],
}));

// --- Mock model-downloader ---
const mockCheckStatus = vi.fn();
const mockDownload = vi.fn();
vi.mock('../../lib/model-downloader', () => ({
  checkModelStatus: (...args: any[]) => mockCheckStatus(...args),
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
  });

  it('GET returns status of all models', async () => {
    mockCheckStatus.mockResolvedValue({ exists: false });

    const route = await importRoute();
    const response = await route.GET();
    const body = await response.json();

    expect(body.models).toHaveLength(3);
    expect(body.models[0]).toHaveProperty('name');
    expect(body.models[0]).toHaveProperty('status');
  });

  it('GET shows downloaded status for existing models', async () => {
    mockCheckStatus.mockResolvedValue({
      exists: true,
      sizeBytes: 4_180_000_000,
      downloadPercent: 100,
    });

    const route = await importRoute();
    const response = await route.GET();
    const body = await response.json();

    expect(body.models[0].status).toBe('downloaded');
  });

  it('POST triggers download of a specific model', async () => {
    mockCheckStatus.mockResolvedValue({ exists: false });
    mockDownload.mockResolvedValue(undefined);

    const route = await importRoute();
    const response = await route.POST(new Request('http://localhost/api/models', {
      method: 'POST',
      body: JSON.stringify({ modelName: 'diffusion_model' }),
    }));
    const body = await response.json();

    expect(body.status).toBe('started');
    expect(mockDownload).toHaveBeenCalled();
  });

  it('POST returns 409 if model already downloaded', async () => {
    mockCheckStatus.mockResolvedValue({
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

  it('POST returns 422 if verification fails after download', async () => {
    mockCheckStatus.mockResolvedValue({ exists: false });
    mockDownload.mockRejectedValue(new Error('Verification failed'));

    const route = await importRoute();
    const response = await route.POST(new Request('http://localhost/api/models', {
      method: 'POST',
      body: JSON.stringify({ modelName: 'vae' }),
    }));

    expect(response.status).toBe(422);
    const body = await response.json();
    expect(body.error).toBeDefined();
  });
});
