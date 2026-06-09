import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('Config API Route', () => {
  let tempDir: string;
  let tempConfigDir: string;
  let tempConfigFile: string;

  beforeEach(() => {
    // Create a real temp directory for testing
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'config-test-'));
    tempConfigDir = path.join(tempDir, '.config');
    tempConfigFile = path.join(tempConfigDir, 'app-config.json');

    // Mock process.cwd() to point to our temp dir
    vi.spyOn(process, 'cwd').mockReturnValue(tempDir);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    // Clean up temp directory
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
    vi.resetModules();
  });

  describe('GET /api/config', () => {
    it('returns default config when no config file exists', async () => {
      const { GET } = await import('../route');
      const response = await GET(null as any);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.config).toHaveProperty('trainingImagesDir');
      expect(data.config).toHaveProperty('outputDir');
      expect(data.config).toHaveProperty('modelsDir');
    });

    it('returns saved config when config file exists', async () => {
      // Create a real config file
      fs.mkdirSync(tempConfigDir, { recursive: true });
      fs.writeFileSync(
        tempConfigFile,
        JSON.stringify({
          trainingImagesDir: '/custom/images',
          outputDir: '/custom/output',
        })
      );

      const { GET } = await import('../route');
      const response = await GET(null as any);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.config.trainingImagesDir).toBe('/custom/images');
      expect(data.config.outputDir).toBe('/custom/output');
    });
  });

  describe('POST /api/config', () => {
    it('saves and returns merged config', async () => {
      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            config: { trainingImagesDir: '/new/path' },
          }),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.config.trainingImagesDir).toBe('/new/path');

      // Verify file was actually written
      expect(fs.existsSync(tempConfigFile)).toBe(true);
      const saved = JSON.parse(fs.readFileSync(tempConfigFile, 'utf-8'));
      expect(saved.trainingImagesDir).toBe('/new/path');
    });

    it('returns 400 for invalid config object', async () => {
      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ invalid: 'data' }),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.error).toBe('Invalid config object');
    });

    it('returns 400 when config is not an object', async () => {
      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: 'string' }),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.error).toBe('Invalid config object');
    });
  });
});
