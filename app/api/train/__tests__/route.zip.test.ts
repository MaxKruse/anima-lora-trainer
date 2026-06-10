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

// --- Mock child_process ---
const mockSpawn = vi.fn();
vi.mock('child_process', () => ({
  default: { spawn: mockSpawn },
  spawn: mockSpawn,
}));

// --- Mock fs ---
const mockExistsSync = vi.fn(() => true);
const mockReadFileSync = vi.fn();
const mockWriteFileSync = vi.fn();
const mockMkdirSync = vi.fn();
const mockReaddirSync = vi.fn(() => ['img1.png', 'img1.txt', 'img2.jpg']);
vi.mock('fs', () => ({
  default: {
    existsSync: (...args: any[]) => mockExistsSync(...args),
    readFileSync: (...args: any[]) => mockReadFileSync(...args),
    writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
    mkdirSync: (...args: any[]) => mockMkdirSync(...args),
    readdirSync: (...args: any[]) => mockReaddirSync(...args),
  },
  existsSync: (...args: any[]) => mockExistsSync(...args),
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  mkdirSync: (...args: any[]) => mockMkdirSync(...args),
  readdirSync: (...args: any[]) => mockReaddirSync(...args),
}));

// --- Mock os ---
vi.mock('os', () => ({
  default: { tmpdir: () => '/tmp' },
  tmpdir: () => '/tmp',
}));

// --- Mock path ---
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

function createValidParams() {
  return {
    networkDim: 32,
    networkAlpha: 16,
    learningRate: 1e-4,
    batchSize: 1,
    epochs: 10,
    optimizer: 'AdamW8Bit',
    scheduler: 'cosine',
    trainingImages: '/path/to/images',
    loraName: 'test-lora',
    mixedPrecision: 'bf16',
    timestepSampling: 'sigmoid',
    gradientCheckpointing: true,
    cacheLatents: true,
    cacheTextEncoder: true,
  };
}

function createMockProcess(exitCode = 0) {
  return {
    stdout: { on: () => {} },
    stderr: { on: () => {} },
    on: (event: string, cb: (code?: number) => void) => {
      if (event === 'close') cb(exitCode);
    },
    kill: () => {},
    pid: 12345,
  };
}

describe('/api/train - training zip', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(true);
    mockMkdirSync.mockReturnValue(undefined);
    mockReaddirSync.mockReturnValue(['img1.png', 'img1.txt', 'img2.jpg']);
    mockSpawn.mockReturnValue(createMockProcess(0));

    const route = await import('../route');
    if (route.__resetJobState) route.__resetJobState();
  });

  it('creates a zip of training data before launching training', async () => {
    mockSpawn.mockReturnValue(createMockProcess(0));

    const route = await import('../route');
    await route.POST(new Request('http://localhost/api/train', {
      method: 'POST',
      body: JSON.stringify(createValidParams()),
    }));

    // Wait for async launchTraining to complete
    await new Promise((r) => setTimeout(r, 100));

    // Should have spawned at least 2 processes: zip + training
    expect(mockSpawn.mock.calls.length).toBeGreaterThanOrEqual(2);

    // One of the calls should be for zip_training_data.py
    const zipCall = mockSpawn.mock.calls.find(
      ([, args]: [string, string[]]) => args.some((a: string) => a.includes('zip_training_data.py'))
    );
    expect(zipCall).toBeDefined();
  });

  it('passes output dir to training command', async () => {
    mockSpawn.mockReturnValue(createMockProcess(0));

    const route = await import('../route');
    await route.POST(new Request('http://localhost/api/train', {
      method: 'POST',
      body: JSON.stringify(createValidParams()),
    }));

    // Wait for async launchTraining to complete
    await new Promise((r) => setTimeout(r, 100));

    // Check that writeFileSync was called to write params JSON
    expect(mockWriteFileSync).toHaveBeenCalled();
    const paramsCall = mockWriteFileSync.mock.calls.find(
      ([filePath]: [string, string]) => {
        return filePath.includes('train-params-');
      }
    );
    expect(paramsCall).toBeDefined();

    // Verify the params include output_dir
    const [, data] = paramsCall;
    const parsed = JSON.parse(data);
    expect(parsed.output_dir).toBeDefined();
    expect(parsed.job_id).toBeDefined();
  });

  it('skips zip creation when source has no images', async () => {
    mockReaddirSync.mockReturnValue(['readme.md']);
    mockSpawn.mockReturnValue(createMockProcess(0));

    const route = await import('../route');
    const response = await route.POST(new Request('http://localhost/api/train', {
      method: 'POST',
      body: JSON.stringify(createValidParams()),
    }));
    const body = await response.json();

    expect(body.status).toBe('started');

    // Should only have 1 spawn call (training, no zip)
    const zipCalls = mockSpawn.mock.calls.filter(
      ([, args]: [string, string[]]) => args.some((a: string) => a.includes('zip_training_data.py'))
    );
    expect(zipCalls.length).toBe(0);
  });
});
