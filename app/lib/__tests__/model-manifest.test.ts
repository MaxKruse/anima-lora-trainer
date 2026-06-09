import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { getModelManifest, getResolvedModelManifest, _clearSizeCache } from '../model-manifest';

describe('getModelManifest', () => {
  it('returns 3 entries for Anima (diffusion model, VAE, text encoder)', () => {
    const manifest = getModelManifest('anima');
    expect(manifest).toHaveLength(3);
  });

  it('each entry has name, hfRepo, hfFile, and localPath', () => {
    const manifest = getModelManifest('anima');
    for (const entry of manifest) {
      expect(entry).toHaveProperty('name');
      expect(entry).toHaveProperty('hfRepo');
      expect(entry).toHaveProperty('hfFile');
      expect(entry).toHaveProperty('localPath');
    }
  });

  it('local paths are under models/anima/', () => {
    const manifest = getModelManifest('anima');
    for (const entry of manifest) {
      expect(entry.localPath).toMatch(/^models\/anima\//);
    }
  });

  it('includes diffusion model entry', () => {
    const manifest = getModelManifest('anima');
    const diffusion = manifest.find(e => e.name === 'diffusion_model');
    expect(diffusion).toBeDefined();
    expect(diffusion!.hfRepo).toBe('circlestone-labs/Anima');
    expect(diffusion!.localPath).toContain('diffusion_models');
  });

  it('includes VAE entry', () => {
    const manifest = getModelManifest('anima');
    const vae = manifest.find(e => e.name === 'vae');
    expect(vae).toBeDefined();
    expect(vae!.hfRepo).toBe('circlestone-labs/Anima');
    expect(vae!.localPath).toContain('vae');
  });

  it('includes text encoder entry', () => {
    const manifest = getModelManifest('anima');
    const te = manifest.find(e => e.name === 'text_encoder');
    expect(te).toBeDefined();
    expect(te!.hfRepo).toBe('circlestone-labs/Anima');
    expect(te!.localPath).toContain('text_encoders');
  });

  it('throws for unknown model type', () => {
    expect(() => getModelManifest('unknown' as any)).toThrow();
  });
});

describe('getResolvedModelManifest', () => {
  beforeEach(() => {
    _clearSizeCache();
    vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('fetches actual file sizes from the HF API', async () => {
    (globalThis.fetch as vi.Mock).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([
        { type: 'file', path: 'split_files/diffusion_models/anima-base-v1.0.safetensors', lfs: { size: 4_182_218_328 } },
        { type: 'file', path: 'split_files/vae/qwen_image_vae.safetensors', lfs: { size: 253_806_246 } },
        { type: 'file', path: 'split_files/text_encoders/qwen_3_06b_base.safetensors', lfs: { size: 1_192_135_096 } },
        { type: 'file', path: 'README.md', size: 9124 },
      ]),
    });

    const manifest = await getResolvedModelManifest('anima');

    expect(manifest).toHaveLength(3);
    expect(manifest[0].expectedSizeBytes).toBe(4_182_218_328);
    expect(manifest[1].expectedSizeBytes).toBe(253_806_246);
    expect(manifest[2].expectedSizeBytes).toBe(1_192_135_096);
  });

  it('falls back to non-lfs size when lfs is absent', async () => {
    (globalThis.fetch as vi.Mock).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([
        { type: 'file', path: 'split_files/diffusion_models/anima-base-v1.0.safetensors', size: 1000 },
      ]),
    });

    const manifest = await getResolvedModelManifest('anima');
    expect(manifest[0].expectedSizeBytes).toBe(1000);
  });

  it('caches results and does not re-fetch within TTL', async () => {
    (globalThis.fetch as vi.Mock).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve([
        { type: 'file', path: 'split_files/diffusion_models/anima-base-v1.0.safetensors', lfs: { size: 999 } },
      ]),
    });

    await getResolvedModelManifest('anima');
    await getResolvedModelManifest('anima');

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });
});
