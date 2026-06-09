import { describe, it, expect, vi, beforeEach } from 'vitest';
import { spawn } from 'child_process';

// Mock spawn
const mockSpawn = vi.fn();
vi.mock('child_process', () => ({
  spawn: mockSpawn,
  default: { spawn: mockSpawn },
}));

vi.mock('fs', () => ({
  default: {
    existsSync: vi.fn(() => false),
    statSync: vi.fn(() => ({ size: 0 })),
    mkdirSync: vi.fn(),
  },
  existsSync: vi.fn(() => false),
  statSync: vi.fn(() => ({ size: 0 })),
  mkdirSync: vi.fn(),
}));

vi.mock('path', () => ({
  default: {
    resolve: (...args: string[]) => args.join('/'),
    join: (...args: string[]) => args.join('/'),
    dirname: (p: string) => p.split('/').slice(0, -1).join('/'),
  },
  resolve: (...args: string[]) => args.join('/'),
  join: (...args: string[]) => args.join('/'),
  dirname: (p: string) => p.split('/').slice(0, -1).join('/'),
}));

async function importDownloader() {
  return await import('../model-downloader');
}

function createMockProcess(
  stdoutData: string,
  stderrData: string,
  exitCode: number
) {
  return {
    stdout: {
      on: (event: string, cb: (data: Buffer) => void) => {
        if (event === 'data' && stdoutData) cb(Buffer.from(stdoutData));
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
      if (event === 'close') cb(exitCode);
    },
    kill: vi.fn(),
  };
}

describe('downloadModel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('calls huggingface-cli with correct args for each model entry', async () => {
    mockSpawn.mockReturnValueOnce(
      createMockProcess('', '', 0)
    );

    const { downloadModel } = await importDownloader();

    await downloadModel({
      name: 'diffusion_model',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/diffusion_models/anima-base-v1.0.safetensors',
      hfPath: 'circlestone-labs/Anima:main:split_files/...',
      localPath: 'models/anima/diffusion_models/anima-base-v1.0.safetensors',
      expectedSizeBytes: 4_180_000_000,
    });

    expect(mockSpawn).toHaveBeenCalled();
    const callArgs = mockSpawn.mock.calls[0];
    const cmd = callArgs[0] as string;
    const args = callArgs[1] as string[];

    expect(cmd).toContain('huggingface-cli') || args.includes('huggingface-cli');
    expect(args).toContain('download');
    expect(args).toContain('circlestone-labs/Anima');
    expect(args).toContain('split_files/diffusion_models/anima-base-v1.0.safetensors');
    expect(args).toContain('--local-dir');
  });

  it('reports progress as download completes', async () => {
    const progressCallback = vi.fn();

    mockSpawn.mockReturnValueOnce(
      createMockProcess('', '', 0)
    );

    const { downloadModel } = await importDownloader();

    await downloadModel({
      name: 'vae',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/vae/qwen_image_vae.safetensors',
      hfPath: 'circlestone-labs/Anima:main:split_files/vae/qwen_image_vae.safetensors',
      localPath: 'models/anima/vae/qwen_image_vae.safetensors',
      expectedSizeBytes: 254_000_000,
    }, progressCallback);

    // Should report 100% on success
    expect(progressCallback).toHaveBeenCalledWith(
      expect.objectContaining({ status: 'completed' })
    );
  });

  it('retries on transient network error up to 3 attempts', async () => {
    // First two attempts fail, third succeeds
    mockSpawn
      .mockReturnValueOnce(createMockProcess('', 'network error', 1))
      .mockReturnValueOnce(createMockProcess('', 'network error', 1))
      .mockReturnValueOnce(createMockProcess('', '', 0));

    const { downloadModel } = await importDownloader();

    await expect(downloadModel({
      name: 'text_encoder',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/text_encoders/qwen_3_06b_base.safetensors',
      hfPath: 'circlestone-labs/Anima:main:split_files/text_encoders/qwen_3_06b_base.safetensors',
      localPath: 'models/anima/text_encoders/qwen_3_06b_base.safetensors',
      expectedSizeBytes: 1_200_000_000,
    })).resolves.not.toThrow();

    expect(mockSpawn).toHaveBeenCalledTimes(3);
  });

  it('throws after 3 failed attempts', async () => {
    mockSpawn
      .mockReturnValueOnce(createMockProcess('', 'network error', 1))
      .mockReturnValueOnce(createMockProcess('', 'network error', 1))
      .mockReturnValueOnce(createMockProcess('', 'network error', 1));

    const { downloadModel } = await importDownloader();

    await expect(downloadModel({
      name: 'vae',
      hfRepo: 'circlestone-labs/Anima',
      hfFile: 'split_files/vae/qwen_image_vae.safetensors',
      hfPath: 'circlestone-labs/Anima:main:split_files/vae/qwen_image_vae.safetensors',
      localPath: 'models/anima/vae/qwen_image_vae.safetensors',
      expectedSizeBytes: 254_000_000,
    })).rejects.toThrow();

    expect(mockSpawn).toHaveBeenCalledTimes(3);
  });
});
