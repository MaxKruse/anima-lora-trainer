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

// --- Mock fs ---
const mockExistsSync = vi.fn(() => false);
const mockReadFileSync = vi.fn();
const mockReaddirSync = vi.fn(() => []);
const fsMocks = {
  existsSync: (...args: any[]) => mockExistsSync(...args),
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  readdirSync: (...args: any[]) => mockReaddirSync(...args),
};
vi.mock('fs', () => ({
  default: fsMocks,
  ...fsMocks,
}));

// --- Mock path ---
const pathMocks = {
  resolve: (...args: string[]) => args.join('/'),
  join: (...args: string[]) => args.join('/'),
};
vi.mock('path', () => ({
  default: pathMocks,
  ...pathMocks,
}));

// --- Mock results-loader ---
const mockLoadResults = vi.fn();
vi.mock('../../../lib/results-loader', () => ({
  loadResults: (...args: any[]) => mockLoadResults(...args),
}));

function createMockRunData(runId: string) {
  return {
    permutations: [
      {
        index: 0,
        params: { 'network-dim': 32, 'learning-rate': 0.0001 },
        status: 'completed',
        output_files: ['lora-000001.safetensors'],
      },
      {
        index: 1,
        params: { 'network-dim': 64, 'learning-rate': 0.001 },
        status: 'completed',
        output_files: ['lora-000002.safetensors'],
      },
    ],
    total: 2,
    completed: 2,
    failed: 0,
  };
}

function createMockEvalData(runId: string) {
  return {
    prompt: 'cat dog bird',
    seed: 42,
    total: 2,
    completed: 2,
    failed: 0,
    results: [
      { perm_name: 'perm-a', status: 'completed', inference_time_ms: 1500 },
      { perm_name: 'perm-b', status: 'completed', inference_time_ms: 1600 },
    ],
  };
}

function createMockResults() {
  return [
    {
      params: { 'network-dim': 32, 'learning-rate': 0.0001 },
      loraFile: 'lora-000001.safetensors',
      imageFile: 'eval_perm-a.png',
      status: 'completed',
      inferenceTimeMs: 1500,
    },
    {
      params: { 'network-dim': 64, 'learning-rate': 0.001 },
      loraFile: 'lora-000002.safetensors',
      imageFile: 'eval_perm-b.png',
      status: 'completed',
      inferenceTimeMs: 1600,
    },
  ];
}

describe('/api/results', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(false);
    mockReaddirSync.mockReturnValue([]);
    mockLoadResults.mockReturnValue(createMockResults());
  });

  it('GET returns list of all completed runs', async () => {
    // Simulate two run directories
    mockReaddirSync.mockReturnValue(['run-001', 'run-002']);
    mockExistsSync.mockImplementation((p: string) => {
      // OUTPUT_DIR exists, and manifest.json exists in each run
      return typeof p === 'string' && (p.includes('output') || p.includes('manifest.json'));
    });

    mockReadFileSync.mockImplementation((p: string) => {
      if (p.includes('run-001')) return JSON.stringify(createMockRunData('run-001'));
      if (p.includes('run-002')) return JSON.stringify(createMockRunData('run-002'));
      return '{}';
    });

    const route = await import('../route');
    const response = await route.GET(new Request('http://localhost/api/results'));
    const body = await response.json();

    expect(body.runs).toHaveLength(2);
    expect(body.runs[0]).toHaveProperty('runId');
    expect(body.runs[0]).toHaveProperty('total');
  });

  it('GET with ?runId=X returns detailed results for that run', async () => {
    mockExistsSync.mockReturnValue(true);
    mockLoadResults.mockReturnValue(createMockResults());

    const route = await import('../route');
    const response = await route.GET(new Request('http://localhost/api/results?runId=run-001'));
    const body = await response.json();

    expect(body.results).toHaveLength(2);
    expect(body.results[0]).toHaveProperty('params');
    expect(body.results[0]).toHaveProperty('loraFile');
    expect(body.results[0]).toHaveProperty('imageFile');
    expect(body.results[0]).toHaveProperty('status');
  });

  it('supports ?sort=param_name query parameter', async () => {
    mockExistsSync.mockReturnValue(true);
    mockLoadResults.mockReturnValue([
      { params: { 'network-dim': 64 }, loraFile: 'a.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 100 },
      { params: { 'network-dim': 32 }, loraFile: 'b.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 200 },
    ]);

    const route = await import('../route');
    const response = await route.GET(
      new Request('http://localhost/api/results?runId=run-001&sort=network-dim')
    );
    const body = await response.json();

    // Should be sorted ascending by network-dim
    expect(body.results[0].params['network-dim']).toBe(32);
    expect(body.results[1].params['network-dim']).toBe(64);
  });

  it('supports ?filter=param_name:value query parameter', async () => {
    mockExistsSync.mockReturnValue(true);
    mockLoadResults.mockReturnValue([
      { params: { 'network-dim': 32, 'optimizer': 'AdamW' }, loraFile: 'a.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 100 },
      { params: { 'network-dim': 64, 'optimizer': 'AdamW' }, loraFile: 'b.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 200 },
      { params: { 'network-dim': 32, 'optimizer': 'Prodigy' }, loraFile: 'c.safetensors', imageFile: null, status: 'completed', inferenceTimeMs: 300 },
    ]);

    const route = await import('../route');
    const response = await route.GET(
      new Request('http://localhost/api/results?runId=run-001&filter=network-dim:32')
    );
    const body = await response.json();

    // Should only include results with network-dim=32
    expect(body.results).toHaveLength(2);
    for (const r of body.results) {
      expect(r.params['network-dim']).toBe(32);
    }
  });
});
