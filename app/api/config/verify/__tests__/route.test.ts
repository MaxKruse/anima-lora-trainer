import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('Verify API Route', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'verify-test-'));
    vi.spyOn(process, 'cwd').mockReturnValue(tempDir);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
    vi.resetModules();
  });

  describe('POST /api/config/verify', () => {
    it('returns exists=true for valid directory', async () => {
      // Create a real directory
      const testDir = path.join(tempDir, 'test-subdir');
      fs.mkdirSync(testDir);

      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: testDir }),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.exists).toBe(true);
      expect(data.isDirectory).toBe(true);
    });

    it('returns exists=true for valid file', async () => {
      // Create a real file
      const testFile = path.join(tempDir, 'test-file.txt');
      fs.writeFileSync(testFile, 'hello');

      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: testFile }),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.exists).toBe(true);
      expect(data.isDirectory).toBe(false);
    });

    it('returns exists=false when path does not exist', async () => {
      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: path.join(tempDir, 'nonexistent') }),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.exists).toBe(false);
    });

    it('returns 400 when path is missing', async () => {
      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.error).toBe('A path string is required');
    });

    it('returns 400 when path is not a string', async () => {
      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: 123 }),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.error).toBe('A path string is required');
    });

    it('resolves relative paths against cwd', async () => {
      // Create a subdirectory
      const subDir = path.join(tempDir, 'relative-test');
      fs.mkdirSync(subDir);

      const { POST } = await import('../route');
      const response = await POST(
        new Request('http://localhost/api/config/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: 'relative-test' }),
        })
      );
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.exists).toBe(true);
      expect(data.path).toBe(subDir);
    });
  });
});
