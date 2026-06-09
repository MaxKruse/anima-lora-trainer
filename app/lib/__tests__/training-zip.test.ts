import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock child_process spawn
const mockSpawn = vi.fn();
vi.mock('child_process', () => ({
  default: { spawn: mockSpawn },
  spawn: mockSpawn,
}));

// Mock fs
const mockExistsSync = vi.fn(() => true);
const mockReaddirSync = vi.fn(() => ['img1.png', 'img1.txt', 'img2.jpg', 'img2.txt']);
vi.mock('fs', () => ({
  default: {
    existsSync: (...args: any[]) => mockExistsSync(...args),
    readdirSync: (...args: any[]) => mockReaddirSync(...args),
  },
  existsSync: (...args: any[]) => mockExistsSync(...args),
  readdirSync: (...args: any[]) => mockReaddirSync(...args),
}));

// Mock path
vi.mock('path', () => ({
  default: {
    resolve: (...args: string[]) => args.join('/'),
    join: (...args: string[]) => args.join('/'),
    extname: (f: string) => f.includes('.') ? `.${f.split('.').pop()!}` : '',
  },
  resolve: (...args: string[]) => args.join('/'),
  join: (...args: string[]) => args.join('/'),
  extname: (f: string) => f.includes('.') ? `.${f.split('.').pop()!}` : '',
}));

function createMockProcess(exitCode = 0) {
  return {
    stdout: { on: () => {} },
    stderr: { on: () => {} },
    on: (event: string, cb: (code?: number) => void) => {
      if (event === 'close') cb(exitCode);
    },
    pid: 99999,
  };
}

describe('createTrainingZip', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(true);
    mockReaddirSync.mockReturnValue(['img1.png', 'img1.txt', 'img2.jpg', 'img2.txt']);
    mockSpawn.mockReturnValue(createMockProcess(0));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('creates a zip of training images and captions', async () => {
    const { createTrainingZip } = await import('../training-zip');

    await createTrainingZip('/source/images', '/output/job-123');

    expect(mockSpawn).toHaveBeenCalled();
    const [cmd, args] = mockSpawn.mock.calls[0] as [string, string[]];
    expect(cmd).toBe('uv');
    expect(args.some(a => a.includes('zip_training_data.py'))).toBe(true);
  });

  it('passes source and destination paths to the script', async () => {
    const { createTrainingZip } = await import('../training-zip');

    await createTrainingZip('/source/images', '/output/job-456');

    const [, args] = mockSpawn.mock.calls[0] as [string, string[]];
    expect(args).toContain('/source/images');
    expect(args).toContain('/output/job-456');
  });

  it('returns the zip file path on success', async () => {
    const { createTrainingZip } = await import('../training-zip');

    const result = await createTrainingZip('/source/images', '/output/job-789');

    expect(result).toContain('training-data.zip');
  });

  it('throws error when source directory does not exist', async () => {
    mockExistsSync.mockReturnValue(false);

    const { createTrainingZip } = await import('../training-zip');

    await expect(createTrainingZip('/nonexistent', '/output/job-123')).rejects.toThrow('Source directory not found');
  });

  it('returns null when source directory has no images', async () => {
    mockExistsSync.mockReturnValue(true);
    mockReaddirSync.mockReturnValue(['readme.md', 'config.json']);

    const { createTrainingZip } = await import('../training-zip');

    const result = await createTrainingZip('/empty/dir', '/output/job-123');

    expect(result).toBeNull();
  });
});
