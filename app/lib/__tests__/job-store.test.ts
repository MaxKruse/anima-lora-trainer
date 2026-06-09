import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import path from 'path';

// Mock fs
const mockReadFileSync = vi.fn();
const mockWriteFileSync = vi.fn();
const mockExistsSync = vi.fn(() => false);
const mockMkdirSync = vi.fn();
const mockReaddirSync = vi.fn(() => []);
vi.mock('fs', () => ({
  default: {
    readFileSync: (...args: any[]) => mockReadFileSync(...args),
    writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
    existsSync: (...args: any[]) => mockExistsSync(...args),
    mkdirSync: (...args: any[]) => mockMkdirSync(...args),
    readdirSync: (...args: any[]) => mockReaddirSync(...args),
  },
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  existsSync: (...args: any[]) => mockExistsSync(...args),
  mkdirSync: (...args: any[]) => mockMkdirSync(...args),
  readdirSync: (...args: any[]) => mockReaddirSync(...args),
}));

// Mock path
vi.mock('path', () => ({
  default: {
    resolve: (...args: string[]) => args.join('/'),
    join: (...args: string[]) => args.join('/'),
    dirname: (_: string) => '/mock',
  },
  resolve: (...args: string[]) => args.join('/'),
  join: (...args: string[]) => args.join('/'),
  dirname: (_: string) => '/mock',
}));

async function importJobStore() {
  vi.resetModules();
  const mod = await import('../job-store');
  return mod;
}

describe('JobStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(false);
    mockMkdirSync.mockReturnValue(undefined);
  });

  afterEach(() => {
    vi.resetModules();
  });

  it('createJob(params) returns unique job ID and stores job', async () => {
    const { JobStore } = await importJobStore();
    const store = new JobStore();

    const jobId = store.createJob({
      loraName: 'test-lora',
      networkDim: 32,
      status: 'pending',
    });

    expect(jobId).toBeDefined();
    expect(typeof jobId).toBe('string');
    expect(mockWriteFileSync).toHaveBeenCalled();
  });

  it('getJob(id) returns job with current status', async () => {
    const { JobStore } = await importJobStore();
    const store = new JobStore();

    const jobId = store.createJob({
      loraName: 'test-lora',
      networkDim: 32,
      status: 'pending',
    });

    const job = store.getJob(jobId);
    expect(job).toBeDefined();
    expect(job!.id).toBe(jobId);
    expect(job!.status).toBe('pending');
    expect(job!.params).toHaveProperty('loraName');
  });

  it('listJobs() returns all jobs', async () => {
    const { JobStore } = await importJobStore();
    const store = new JobStore();

    store.createJob({ loraName: 'lora-1', status: 'pending' });
    store.createJob({ loraName: 'lora-2', status: 'running' });
    store.createJob({ loraName: 'lora-3', status: 'completed' });

    const jobs = store.listJobs();
    expect(jobs).toHaveLength(3);
    expect(jobs[0]).toHaveProperty('id');
    expect(jobs[0]).toHaveProperty('status');
  });

  it('job state persists to file (survives process restart)', async () => {
    // First "process" — create a job
    const { JobStore: JobStore1 } = await importJobStore();
    const store1 = new JobStore1();
    const jobId = store1.createJob({
      loraName: 'persistent-lora',
      networkDim: 64,
      status: 'running',
    });

    // Verify writeFileSync was called (persistence)
    expect(mockWriteFileSync).toHaveBeenCalledWith(
      expect.stringContaining(jobId),
      expect.stringContaining('persistent-lora')
    );
  });

  it('loading from file restores all jobs', async () => {
    // Simulate existing job files
    const job1Data = {
      id: 'job-001',
      params: { loraName: 'restored-lora', networkDim: 128 },
      status: 'completed',
      createdAt: '2024-01-01T00:00:00.000Z',
    };
    const job2Data = {
      id: 'job-002',
      params: { loraName: 'another-lora', networkDim: 64 },
      status: 'failed',
      createdAt: '2024-01-02T00:00:00.000Z',
    };

    mockExistsSync.mockReturnValue(true);
    mockReaddirSync.mockReturnValue(['job-001.json', 'job-002.json']);
    mockReadFileSync
      .mockReturnValueOnce(JSON.stringify(job1Data))
      .mockReturnValueOnce(JSON.stringify(job2Data));

    const { JobStore } = await importJobStore();
    const store = new JobStore();

    const jobs = store.listJobs();
    expect(jobs).toHaveLength(2);
    expect(jobs.find((j: any) => j.id === 'job-001')).toBeDefined();
    expect(jobs.find((j: any) => j.id === 'job-002')).toBeDefined();
    expect(store.getJob('job-001')!.status).toBe('completed');
    expect(store.getJob('job-002')!.status).toBe('failed');
  });
});
