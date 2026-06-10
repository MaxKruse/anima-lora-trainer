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
const mockExistsSync = vi.fn(() => false);
const mockReadFileSync = vi.fn();
const mockWriteFileSync = vi.fn();
const mockMkdirSync = vi.fn();
const mockRmSync = vi.fn();
const mockUnlinkSync = vi.fn();
vi.mock('fs', () => ({
  default: {
    existsSync: (...args: any[]) => mockExistsSync(...args),
    readFileSync: (...args: any[]) => mockReadFileSync(...args),
    writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
    mkdirSync: (...args: any[]) => mockMkdirSync(...args),
    rmSync: (...args: any[]) => mockRmSync(...args),
    unlinkSync: (...args: any[]) => mockUnlinkSync(...args),
  },
  existsSync: (...args: any[]) => mockExistsSync(...args),
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  mkdirSync: (...args: any[]) => mockMkdirSync(...args),
  rmSync: (...args: any[]) => mockRmSync(...args),
  unlinkSync: (...args: any[]) => mockUnlinkSync(...args),
}));

// --- Mock os ---
vi.mock('os', () => ({
  default: { tmpdir: () => '/tmp' },
  tmpdir: () => '/tmp',
}));

// --- Mock training-zip ---
// Path is relative to the route.ts file, not this test file
vi.mock('../../../lib/training-zip', () => ({
  createTrainingZip: vi.fn().mockResolvedValue(null),
}));

// --- Mock path ---
vi.mock('path', () => ({
  default: {
    resolve: (...args: string[]) => args.join('/'),
    join: (...args: string[]) => args.join('/'),
  },
  resolve: (...args: string[]) => args.join('/'),
  join: (...args: string[]) => args.join('/'),
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

function createMockProcess() {
  return {
    stdout: {
      on: () => {},
      removeAllListeners: () => {},
    },
    stderr: {
      on: () => {},
      removeAllListeners: () => {},
    },
    on: () => {},
    kill: () => {},
    pid: 12345,
  };
}

describe('/api/train', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(false);
    mockMkdirSync.mockReturnValue(undefined);

    // Reset in-memory job state between tests
    const route = await import('../route');
    if (route.__resetJobState) route.__resetJobState();
  });

  it('POST with valid params returns { jobId, status: "started" }', async () => {
    mockSpawn.mockReturnValue(createMockProcess());

    const route = await import('../route');
    const response = await route.POST(new Request('http://localhost/api/train', {
      method: 'POST',
      body: JSON.stringify(createValidParams()),
    }));
    const body = await response.json();

    expect(body.status).toBe('started');
    expect(body).toHaveProperty('jobId');
    expect(typeof body.jobId).toBe('string');
  });

  it('POST with invalid params returns 400 with validation errors', async () => {
    const route = await import('../route');
    const response = await route.POST(new Request('http://localhost/api/train', {
      method: 'POST',
      body: JSON.stringify({ networkDim: -1, learningRate: 0 }),
    }));

    expect(response.status).toBe(400);
    const body = await response.json();
    expect(body.error).toBeDefined();
  });

  it('POST when another job is running returns 409', async () => {
    mockSpawn.mockReturnValue(createMockProcess());

    const route = await import('../route');

    // Start first job
    const firstResponse = await route.POST(new Request('http://localhost/api/train', {
      method: 'POST',
      body: JSON.stringify(createValidParams()),
    }));
    const firstBody = await firstResponse.json();
    expect(firstBody.status).toBe('started');

    // Try to start a second job (same module instance, so job state persists)
    const secondResponse = await route.POST(new Request('http://localhost/api/train', {
      method: 'POST',
      body: JSON.stringify(createValidParams()),
    }));

    expect(secondResponse.status).toBe(409);
    const secondBody = await secondResponse.json();
    expect(secondBody.error).toBeDefined();
  });

  it('Job ID is unique (timestamp + random suffix)', async () => {
    mockSpawn.mockReturnValue(createMockProcess());

    const route = await import('../route');

    const resp1 = await route.POST(new Request('http://localhost/api/train', {
      method: 'POST',
      body: JSON.stringify(createValidParams()),
    }));
    const body1 = await resp1.json();

    // Verify job ID format: job-{timestamp}-{random}
    expect(body1.jobId).toMatch(/^job-\d+-[a-z0-9]+$/);
    const parts = body1.jobId.split('-');
    expect(parts.length).toBeGreaterThanOrEqual(3);
    // Timestamp part should be a positive number
    expect(Number(parts[1])).toBeGreaterThan(0);
  });
});
