import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.resolve(process.cwd());
const JOBS_DIR = path.join(PROJECT_ROOT, 'jobs');

/**
 * JobController — manages pause/resume/cancel signals for training jobs.
 *
 * Writes signal files that the matrix trainer script checks between
 * permutations to decide whether to pause or stop.
 */
export class JobController {
  private processes = new Map<string, any>();

  /**
   * Register a child process for a job.
   */
  registerProcess(jobId: string, proc: any): void {
    this.processes.set(jobId, proc);
  }

  /**
   * Pause a job by writing a pause signal file.
   */
  pause(jobId: string): void {
    const signalPath = path.join(JOBS_DIR, `${jobId}.pause`);
    fs.writeFileSync(signalPath, new Date().toISOString());
  }

  /**
   * Resume a job by removing the pause signal file.
   */
  resume(jobId: string): void {
    const signalPath = path.join(JOBS_DIR, `${jobId}.pause`);
    if (fs.existsSync(signalPath)) {
      fs.unlinkSync(signalPath);
    }
  }

  /**
   * Cancel a job by writing a cancel signal and terminating the process.
   */
  cancel(jobId: string): void {
    const cancelPath = path.join(JOBS_DIR, `${jobId}.cancel`);
    fs.writeFileSync(cancelPath, new Date().toISOString());

    const proc = this.processes.get(jobId);
    if (proc) {
      proc.kill('SIGTERM');
      this.processes.delete(jobId);
    }
  }

  /**
   * Check if a job is currently paused.
   */
  isPaused(jobId: string): boolean {
    const signalPath = path.join(JOBS_DIR, `${jobId}.pause`);
    return fs.existsSync(signalPath);
  }

  /**
   * Check if a job has been cancelled.
   */
  isCancelled(jobId: string): boolean {
    const signalPath = path.join(JOBS_DIR, `${jobId}.cancel`);
    return fs.existsSync(signalPath);
  }

  /**
   * Remove all signal files for a job (cleanup after completion).
   */
  cleanup(jobId: string): void {
    const pausePath = path.join(JOBS_DIR, `${jobId}.pause`);
    const cancelPath = path.join(JOBS_DIR, `${jobId}.cancel`);

    if (fs.existsSync(pausePath)) fs.unlinkSync(pausePath);
    if (fs.existsSync(cancelPath)) fs.unlinkSync(cancelPath);
    this.processes.delete(jobId);
  }
}
