import { describe, it, expect } from 'vitest';
import { getModelManifest } from '../model-manifest';

describe('getModelManifest', () => {
  it('returns 3 entries for Anima (diffusion model, VAE, text encoder)', () => {
    const manifest = getModelManifest('anima');
    expect(manifest).toHaveLength(3);
  });

  it('each entry has hfPath, localPath, and expectedSizeBytes', () => {
    const manifest = getModelManifest('anima');
    for (const entry of manifest) {
      expect(entry).toHaveProperty('name');
      expect(entry).toHaveProperty('hfPath');
      expect(entry).toHaveProperty('localPath');
      expect(entry).toHaveProperty('expectedSizeBytes');
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
    expect(diffusion!.hfPath).toContain('circlestone-labs/Anima');
    expect(diffusion!.localPath).toContain('diffusion_models');
  });

  it('includes VAE entry', () => {
    const manifest = getModelManifest('anima');
    const vae = manifest.find(e => e.name === 'vae');
    expect(vae).toBeDefined();
    expect(vae!.hfPath).toContain('circlestone-labs/Anima');
    expect(vae!.localPath).toContain('vae');
  });

  it('includes text encoder entry', () => {
    const manifest = getModelManifest('anima');
    const te = manifest.find(e => e.name === 'text_encoder');
    expect(te).toBeDefined();
    expect(te!.hfPath).toContain('circlestone-labs/Anima');
    expect(te!.localPath).toContain('text_encoders');
  });

  it('throws for unknown model type', () => {
    expect(() => getModelManifest('unknown' as any)).toThrow();
  });
});
