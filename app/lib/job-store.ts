import fs from 'fs';
import path from 'path';

const JOBS_DIR = path.join(process.cwd(), 'jobs');

export interface JobRecord {
  id: string;
  params: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  createdAt: string;
  updatedAt?: string;
  error?: string;
  pid?: number;
}

/**
 * In-memory + file-based job state tracker.
 *
 * Each job is stored as a JSON file in the jobs/ directory.
 * The store also keeps an in-memory index for fast lookups.
 */
export class JobStore {
  private jobs: Map<string, JobRecord> = new Map();

  constructor() {
    this.loadFromFile();
  }

  /**
   * Create a new job and persist it to disk.
   */
  createJob(params: Record<string, any>): string {
    const id = this.generateJobId();
    const job: JobRecord = {
      id,
      params,
      status: 'pending',
      createdAt: new Date().toISOString(),
    };

    this.jobs.set(id, job);
    this.persistJob(job);
    return id;
  }

  /**
   * Get a job by ID.
   */
  getJob(id: string): JobRecord | undefined {
    return this.jobs.get(id);
  }

  /**
   * List all jobs.
   */
  listJobs(): JobRecord[] {
    return Array.from(this.jobs.values());
  }

  /**
   * Update a job's status.
   */
  updateStatus(id: string, status: JobRecord['status']): boolean {
    const job = this.jobs.get(id);
    if (!job) return false;

    job.status = status;
    job.updatedAt = new Date().toISOString();
    this.persistJob(job);
    return true;
  }

  /**
   * Update a job's error message.
   */
  updateError(id: string, error: string): boolean {
    const job = this.jobs.get(id);
    if (!job) return false;

    job.error = error;
    job.updatedAt = new Date().toISOString();
    this.persistJob(job);
    return true;
  }

  /**
   * Update a job's PID.
   */
  updatePid(id: string, pid: number): boolean {
    const job = this.jobs.get(id);
    if (!job) return false;

    job.pid = pid;
    this.persistJob(job);
    return true;
  }

  /**
   * Check if a process is alive by PID.
   */
  static isProcessAlive(pid: number): boolean {
    try {
      process.kill(pid, 0); // Sends no signal, just checks existence
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Recover jobs after server restart.
   * Marks dead processes as failed, returns PIDs of still-running processes.
   */
  recoverJobs(): Map<string, number> {
    const alivePids = new Map<string, number>();

    for (const job of this.listJobs()) {
      if (job.status === 'running' && job.pid) {
        if (JobStore.isProcessAlive(job.pid)) {
          alivePids.set(job.id, job.pid);
        } else {
          this.updateStatus(job.id, 'failed');
          this.updateError(job.id, 'Process died (server restart or crash)');
        }
      }
    }

    return alivePids;
  }

  /**
   * Delete a job.
   */
  deleteJob(id: string): boolean {
    const job = this.jobs.get(id);
    if (!job) return false;

    this.jobs.delete(id);
    const filePath = path.join(JOBS_DIR, `${id}.json`);
    try {
      if (fs.existsSync(filePath)) {
        fs.writeFileSync(filePath, '');
      }
    } catch {
      // Ignore file deletion errors
    }
    return true;
  }

  /**
   * Generate a unique job ID using timestamp + random suffix.
   */
  private generateJobId(): string {
    const timestamp = Date.now();
    const randomSuffix = Math.random().toString(36).substring(2, 8);
    return `job-${timestamp}-${randomSuffix}`;
  }

  /**
   * Persist a single job to disk.
   */
  private persistJob(job: JobRecord): void {
    try {
      if (!fs.existsSync(JOBS_DIR)) {
        fs.mkdirSync(JOBS_DIR, { recursive: true });
      }
      const filePath = path.join(JOBS_DIR, `${job.id}.json`);
      fs.writeFileSync(filePath, JSON.stringify(job, null, 2));
    } catch {
      // Ignore persistence errors
    }
  }

  /**
   * Load all jobs from disk on initialization.
   */
  private loadFromFile(): void {
    try {
      if (!fs.existsSync(JOBS_DIR)) return;

      const files = fs.readdirSync(JOBS_DIR).filter((f) => f.endsWith('.json'));

      for (const file of files) {
        const filePath = path.join(JOBS_DIR, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        if (!content.trim()) continue;

        try {
          const job = JSON.parse(content) as JobRecord;
          if (job.id && job.status) {
            this.jobs.set(job.id, job);
          }
        } catch {
          // Skip invalid JSON files
        }
      }
    } catch {
      // Ignore loading errors
    }
  }
}

// Singleton instance for API routes
let _store: JobStore | null = null;

export function getJobStore(): JobStore {
  if (!_store) {
    _store = new JobStore();
  }
  return _store;
}

/** Reset the singleton. Used for testing. */
export function __resetJobStore(): void {
  _store = null;
}
