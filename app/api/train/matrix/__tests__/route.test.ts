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
const mockWriteFileSync = vi.fn();
const mockMkdirSync = vi.fn();
vi.mock('fs', () => ({
  default: {
    existsSync: (...args: any[]) => mockExistsSync(...args),
    writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
    mkdirSync: (...args: any[]) => mockMkdirSync(...args),
  },
  existsSync: (...args: any[]) => mockExistsSync(...args),
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  mkdirSync: (...args: any[]) => mockMkdirSync(...args),
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

// --- Mock job-store ---
const mockGetJobStore = vi.fn();
vi.mock('../../../../lib/job-store', () => ({
  getJobStore: () => mockGetJobStore(),
  __resetJobStore: () => {},
}));

async function importRoute() {
  return await import('../route');
}

function createMockStore() {
  const jobs: any[] = [];
  return {
    listJobs: () => jobs,
    getJob: (id: string) => jobs.find((j) => j.id === id),
    createJob: (params: any) => {
      const id = `job-${Date.now()}`;
      jobs.push({ id, ...params, status: 'pending' });
      return id;
    },
    updateStatus: (id: string, status: string) => {
      const job = jobs.find((j) => j.id === id);
      if (job) job.status = status;
    },
  };
}

function createMockProcess() {
  return {
    stdout: { on: () => {}, removeAllListeners: () => {} },
    stderr: { on: () => {}, removeAllListeners: () => {} },
    on: () => {},
    kill: () => {},
    pid: 12345,
  };
}

describe('/api/train/matrix', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetJobStore.mockReturnValue(createMockStore());
    mockSpawn.mockReturnValue(createMockProcess());
  });

  it('POST with valid matrix params returns { jobId, permutationCount, status: "started" }', async () => {
    const route = await importRoute();
    const response = await route.POST(new Request('http://localhost/api/train/matrix', {
      method: 'POST',
      body: JSON.stringify({
        paramRanges: {
          network_dim: '8,16,32',
          network_alpha: '1,4,8',
          learning_rate: '1e-4,5e-4',
        },
        baseParams: {
          trainingImages: '/path/to/images',
          loraName: 'matrix-lora',
          epochs: 10,
          batchSize: 1,
          optimizer: 'AdamW8Bit',
          scheduler: 'cosine',
          mixedPrecision: 'bf16',
          timestepSampling: 'sigmoid',
        },
      }),
    }));
    const body = await response.json();

    expect(body.status).toBe('started');
    expect(body).toHaveProperty('jobId');
    expect(body.permutationCount).toBe(3 * 3 * 2); // 18 permutations
  });

  it('rejects params that would produce 0 permutations', async () => {
    const route = await importRoute();
    const response = await route.POST(new Request('http://localhost/api/train/matrix', {
      method: 'POST',
      body: JSON.stringify({
        paramRanges: {},
        baseParams: {},
      }),
    }));

    expect(response.status).toBe(400);
    const body = await response.json();
    expect(body.error).toBeDefined();
  });

  it('returns 400 if any parameter range is empty', async () => {
    const route = await importRoute();
    const response = await route.POST(new Request('http://localhost/api/train/matrix', {
      method: 'POST',
      body: JSON.stringify({
        paramRanges: {
          network_dim: '',
        },
        baseParams: {},
      }),
    }));

    expect(response.status).toBe(400);
    const body = await response.json();
    expect(body.error).toBeDefined();
  });
});
