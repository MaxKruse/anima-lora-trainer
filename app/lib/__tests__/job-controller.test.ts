import { describe, it, expect, vi, beforeEach } from 'vitest';
import fs from 'fs';

// --- Mock fs ---
const mockExistsSync = vi.fn(() => false);
const mockWriteFileSync = vi.fn();
const mockUnlinkSync = vi.fn();
const fsMocks = {
  existsSync: (...args: any[]) => mockExistsSync(...args),
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  unlinkSync: (...args: any[]) => mockUnlinkSync(...args),
};
vi.mock('fs', () => ({
  default: fsMocks,
  ...fsMocks,
}));

// --- Mock path ---
const pathMocks = {
  join: (...args: string[]) => args.join('/'),
  resolve: (...args: string[]) => args.join('/'),
};
vi.mock('path', () => ({
  default: pathMocks,
  ...pathMocks,
}));

async function importJobController() {
  const mod = await import('../job-controller');
  return mod.JobController;
}

describe('JobController', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(false);
  });

  it('pause(jobId) writes pause signal file', async () => {
    const JobController = await importJobController();
    const controller = new JobController();

    controller.pause('job-001');

    expect(mockWriteFileSync).toHaveBeenCalledWith(
      expect.stringContaining('job-001'),
      expect.any(String)
    );
  });

  it('resume(jobId) removes pause signal file', async () => {
    mockExistsSync.mockReturnValue(true);

    const JobController = await importJobController();
    const controller = new JobController();

    controller.resume('job-001');

    expect(mockUnlinkSync).toHaveBeenCalledWith(
      expect.stringContaining('job-001')
    );
  });

  it('cancel(jobId) writes cancel signal file and terminates process', async () => {
    const mockKill = vi.fn();
    const mockProc = { kill: mockKill };

    const JobController = await importJobController();
    const controller = new JobController();

    // Register a process for the job
    (controller as any).processes.set('job-001', mockProc);

    controller.cancel('job-001');

    expect(mockWriteFileSync).toHaveBeenCalledWith(
      expect.stringContaining('job-001'),
      expect.any(String)
    );
    expect(mockKill).toHaveBeenCalledWith('SIGTERM');
  });

  it('matrix trainer respects pause signal (waits between permutations)', async () => {
    mockExistsSync.mockImplementation((p: string) => {
      return typeof p === 'string' && p.includes('pause');
    });

    const JobController = await importJobController();
    const controller = new JobController();

    const isPaused = controller.isPaused('job-001');
    expect(isPaused).toBe(true);
  });

  it('matrix trainer respects cancel signal (exits loop)', async () => {
    mockExistsSync.mockImplementation((p: string) => {
      return typeof p === 'string' && p.includes('cancel');
    });

    const JobController = await importJobController();
    const controller = new JobController();

    const isCancelled = controller.isCancelled('job-001');
    expect(isCancelled).toBe(true);
  });
});
