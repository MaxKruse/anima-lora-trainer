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

// --- Mock job-store ---
const mockGetJobStore = vi.fn();
vi.mock('../../../lib/job-store', () => ({
  getJobStore: (...args: any[]) => mockGetJobStore(...args),
  __resetJobStore: () => {},
}));

async function importRoute() {
  return await import('../route');
}

function createMockStore(jobs: any[] = []) {
  return {
    listJobs: () => jobs,
    getJob: (id: string) => jobs.find((j) => j.id === id),
  };
}

describe('/api/jobs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('GET returns array of all jobs', async () => {
    const jobs = [
      { id: 'job-001', status: 'running', params: { loraName: 'test-1' }, createdAt: '2024-01-01T00:00:00.000Z' },
      { id: 'job-002', status: 'completed', params: { loraName: 'test-2' }, createdAt: '2024-01-02T00:00:00.000Z' },
    ];
    mockGetJobStore.mockReturnValue(createMockStore(jobs));

    const route = await importRoute();
    const response = await route.GET(new Request('http://localhost/api/jobs'));
    const body = await response.json();

    expect(body.jobs).toHaveLength(2);
    expect(body.jobs[0]).toHaveProperty('id');
    expect(body.jobs[0]).toHaveProperty('status');
  });

  it('GET with query param ?id=X returns single job', async () => {
    const jobs = [
      { id: 'job-001', status: 'running', params: { loraName: 'test-1' }, createdAt: '2024-01-01T00:00:00.000Z' },
      { id: 'job-002', status: 'completed', params: { loraName: 'test-2' }, createdAt: '2024-01-02T00:00:00.000Z' },
    ];
    mockGetJobStore.mockReturnValue(createMockStore(jobs));

    const route = await importRoute();
    const response = await route.GET(new Request('http://localhost/api/jobs?id=job-002'));
    const body = await response.json();

    expect(body.id).toBe('job-002');
    expect(body.status).toBe('completed');
  });

  it('returns empty array when no jobs exist', async () => {
    mockGetJobStore.mockReturnValue(createMockStore([]));

    const route = await importRoute();
    const response = await route.GET(new Request('http://localhost/api/jobs'));
    const body = await response.json();

    expect(body.jobs).toHaveLength(0);
  });
});
