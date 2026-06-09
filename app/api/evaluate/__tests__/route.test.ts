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
  spawn: mockSpawn,
  default: { spawn: mockSpawn },
}));

// --- Mock fs ---
const mockExistsSync = vi.fn(() => false);
const mockReadFileSync = vi.fn();
const mockWriteFileSync = vi.fn();
const mockMkdirSync = vi.fn();
const mockReaddirSync = vi.fn(() => []);
const fsMocks = {
  existsSync: (...args: any[]) => mockExistsSync(...args),
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  mkdirSync: (...args: any[]) => mockMkdirSync(...args),
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

function createMockProcess() {
  return {
    stdout: { on: () => {}, removeAllListeners: () => {} },
    stderr: { on: () => {}, removeAllListeners: () => {} },
    on: () => {},
    kill: () => {},
    pid: 99999,
  };
}

describe('/api/evaluate', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(false);
    mockMkdirSync.mockReturnValue(undefined);
    mockReaddirSync.mockReturnValue([]);
    mockSpawn.mockReturnValue(createMockProcess());

    // Reset evaluation state between tests
    const route = await import('../route');
    if (route.__resetEvalState) route.__resetEvalState();
  });

  it('POST with valid run ID starts evaluation', async () => {
    // Simulate that the run directory exists
    mockExistsSync.mockImplementation((p: string) => {
      return typeof p === 'string' && p.includes('output');
    });

    const route = await import('../route');
    const response = await route.POST(new Request('http://localhost/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({ runId: 'run-001' }),
    }));
    const body = await response.json();

    expect(body.status).toBe('started');
    expect(body).toHaveProperty('runId');
    expect(body.runId).toBe('run-001');
  });

  it('Returns 404 for non-existent run ID', async () => {
    // Simulate that the run directory does NOT exist
    mockExistsSync.mockReturnValue(false);

    const route = await import('../route');
    const response = await route.POST(new Request('http://localhost/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({ runId: 'nonexistent-run' }),
    }));

    expect(response.status).toBe(404);
    const body = await response.json();
    expect(body.error).toBeDefined();
  });

  it('Returns 409 if evaluation already running for this run', async () => {
    // Simulate that the run directory exists
    mockExistsSync.mockImplementation((p: string) => {
      return typeof p === 'string' && p.includes('output');
    });

    const route = await import('../route');

    // Start first evaluation
    const firstResponse = await route.POST(new Request('http://localhost/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({ runId: 'run-001' }),
    }));
    const firstBody = await firstResponse.json();
    expect(firstBody.status).toBe('started');

    // Try to start a second evaluation for the same run
    const secondResponse = await route.POST(new Request('http://localhost/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({ runId: 'run-001' }),
    }));

    expect(secondResponse.status).toBe(409);
    const secondBody = await secondResponse.json();
    expect(secondBody.error).toBeDefined();
  });

  it('Returns evaluation results when complete', async () => {
    // Simulate that the run directory exists AND evaluation.json exists
    mockExistsSync.mockImplementation((p: string) => {
      return typeof p === 'string' && (p.includes('output') || p.includes('evaluation.json'));
    });

    // Simulate reading evaluation.json
    const mockEvalData = {
      prompt: 'cat dog bird',
      seed: 42,
      total: 3,
      completed: 3,
      failed: 0,
      results: [
        { perm_name: 'perm-a', status: 'completed', inference_time_ms: 1500 },
        { perm_name: 'perm-b', status: 'completed', inference_time_ms: 1600 },
        { perm_name: 'perm-c', status: 'completed', inference_time_ms: 1400 },
      ],
    };
    mockReadFileSync.mockReturnValue(JSON.stringify(mockEvalData));

    const route = await import('../route');

    // GET results for a completed evaluation
    const response = await route.GET(new Request('http://localhost/api/evaluate?runId=run-001'));
    const body = await response.json();

    expect(body.prompt).toBe('cat dog bird');
    expect(body.total).toBe(3);
    expect(body.results).toHaveLength(3);
  });
});
