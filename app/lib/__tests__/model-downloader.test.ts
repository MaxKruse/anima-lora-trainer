import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock child_process
const mockSpawn = vi.fn();
const mockExecSync = vi.fn();
vi.mock('child_process', () => ({
  spawn: mockSpawn,
  execSync: mockExecSync,
  default: { spawn: mockSpawn, execSync: mockExecSync },
}));

vi.mock('fs', () => ({
  default: {
    existsSync: vi.fn(() => false),
    statSync: vi.fn(() => ({ size: 0 })),
    mkdirSync: vi.fn(),
    mkdtempSync: vi.fn(() => '/tmp/hf-dl-test'),
    writeFileSync: vi.fn(),
    unlinkSync: vi.fn(),
    rmdirSync: vi.fn(),
  },
  existsSync: vi.fn(() => false),
  statSync: vi.fn(() => ({ size: 0 })),
  mkdirSync: vi.fn(),
  mkdtempSync: vi.fn(() => '/tmp/hf-dl-test'),
  writeFileSync: vi.fn(),
  unlinkSync: vi.fn(),
  rmdirSync: vi.fn(),
}));

vi.mock('os', () => ({
  default: { tmpdir: () => '/tmp' },
  tmpdir: () => '/tmp',
}));

vi.mock('path', () => ({
  default: {
    join: (...args: string[]) => args.join('/'),
    basename: (path: string) => path.split('/').pop() || path,
  },
  join: (...args: string[]) => args.join('/'),
  basename: (path: string) => path.split('/').pop() || path,
}));

async function importDownloader() {
  return await import('../model-downloader');
}

function createMockProcess(
  stdoutLines: string[],
  stderrData: string,
  exitCode: number,
  delayMs = 0
) {
  return {
    stdout: {
      on: (event: string, cb: (data: Buffer) => void) => {
        if (event === 'data') {
          const timer = setTimeout(() => {
            for (const line of stdoutLines) {
              cb(Buffer.from(line + '\n'));
            }
          }, delayMs);
          // Store timer for cleanup
          (cb as any)._timer = timer;
        }
      },
      removeAllListeners: () => {},
    },
    stderr: {
      on: (event: string, cb: (data: Buffer) => void) => {
        if (event === 'data' && stderrData) cb(Buffer.from(stderrData));
      },
      removeAllListeners: () => {},
    },
    on: (event: string, cb: (code: number | null) => void) => {
      if (event === 'close') {
        setTimeout(() => cb(exitCode), delayMs + 10);
      }
    },
    kill: vi.fn(),
  };
}

describe('downloadModel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('spawns Python with hf_hub_download script', async () => {
    mockSpawn.mockReturnValueOnce(
      createMockProcess(['{"type": "progress", "percent": 100}', '{"type": "done", "path": "/cached/file"}'], '', 0)
    );

    const { downloadModel } = await importDownloader();

    await downloadModel({
      name: 'diffusion_model',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/diffusion_models/anima-base-v1.0.safetensors',
    });

    expect(mockSpawn).toHaveBeenCalled();
    const callArgs = mockSpawn.mock.calls[0];
    const cmd = callArgs[0] as string;
    const args = callArgs[1] as string[];

    expect(cmd).toBe('uv');
    expect(args).toContain('run');
    expect(args).toContain('python');
    // The URL and dest path are passed as args to the Python script
    const urlArg = args.find((a: string) => a.startsWith('https://'));
    expect(urlArg).toContain('circlestone-labs/Anima');
    expect(urlArg).toContain('split_files/diffusion_models/anima-base-v1.0.safetensors');
  });

  it('reports progress from Python stdout JSON lines', async () => {
    const progressCallback = vi.fn();

    mockSpawn.mockReturnValueOnce(
      createMockProcess([
        '{"type": "progress", "percent": 10}',
        '{"type": "progress", "percent": 50}',
        '{"type": "progress", "percent": 100}',
        '{"type": "done", "path": "/cached/file"}',
      ], '', 0)
    );

    const { downloadModel } = await importDownloader();

    await downloadModel({
      name: 'vae',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/vae/qwen_image_vae.safetensors',
    }, progressCallback);

    // Should report progress updates
    expect(progressCallback).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'vae', status: 'downloading', progress: 10 })
    );
    expect(progressCallback).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'vae', status: 'downloading', progress: 50 })
    );
    // Final completion
    expect(progressCallback).toHaveBeenCalledWith(
      expect.objectContaining({ status: 'completed' })
    );
  });

  it('retries on transient error up to 3 attempts', async () => {
    mockSpawn
      .mockReturnValueOnce(createMockProcess(['{"type": "error", "message": "network error"}'], '', 1))
      .mockReturnValueOnce(createMockProcess(['{"type": "error", "message": "network error"}'], '', 1))
      .mockReturnValueOnce(createMockProcess(['{"type": "progress", "percent": 100}', '{"type": "done", "path": "/cached"}'], '', 0));

    const { downloadModel } = await importDownloader();

    await expect(downloadModel({
      name: 'text_encoder',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/text_encoders/qwen_3_06b_base.safetensors',
    })).resolves.not.toThrow();

    expect(mockSpawn).toHaveBeenCalledTimes(3);
  });

  it('throws after 3 failed attempts', async () => {
    mockSpawn
      .mockReturnValueOnce(createMockProcess(['{"type": "error", "message": "network error"}'], '', 1))
      .mockReturnValueOnce(createMockProcess(['{"type": "error", "message": "network error"}'], '', 1))
      .mockReturnValueOnce(createMockProcess(['{"type": "error", "message": "network error"}'], '', 1));

    const { downloadModel } = await importDownloader();

    await expect(downloadModel({
      name: 'vae',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/vae/qwen_image_vae.safetensors',
    })).rejects.toThrow();

    expect(mockSpawn).toHaveBeenCalledTimes(3);
  });
});
