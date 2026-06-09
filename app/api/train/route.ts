import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { trainingSchema, type TrainingParams } from '../../lib/training-schema';
import { createTrainingZip } from '../../lib/training-zip';
import { getJobStore } from '../../lib/job-store';

const PROJECT_ROOT = path.resolve(process.cwd());
const CONFIG_DIR = path.join(PROJECT_ROOT, '.config');
const CONFIG_FILE = path.join(CONFIG_DIR, 'app-config.json');

/**
 * In-memory state to track currently running jobs and their child processes.
 */
const activeJobs = new Map<string, { params: TrainingParams; proc: any }>();

/**
 * Reset active jobs state. Used for testing.
 */
export function __resetJobState(): void {
  activeJobs.clear();
}

/**
 * Recover jobs after server restart.
 * Checks running jobs for alive PIDs and marks dead ones as failed.
 */
function recoverJobs(): void {
  const store = getJobStore();
  const alivePids = store.recoverJobs();

  // Re-register alive jobs in activeJobs map
  for (const [jobId, pid] of alivePids) {
    const job = store.getJob(jobId);
    if (job) {
      activeJobs.set(jobId, { params: job.params as TrainingParams, proc: null });
      console.log(`[train:${jobId}] recovered running job (pid: ${pid})`);
    }
  }
}

// Run recovery on module load
recoverJobs();

/**
 * Load the app config to get user's output directory setting.
 */
function loadConfig(): { outputDir: string } {
  try {
    if (!fs.existsSync(CONFIG_FILE)) {
      return { outputDir: path.join(PROJECT_ROOT, 'output') };
    }
    const raw = fs.readFileSync(CONFIG_FILE, 'utf-8');
    const saved = JSON.parse(raw);
    return {
      outputDir: saved.outputDir || path.join(PROJECT_ROOT, 'output'),
    };
  } catch {
    return { outputDir: path.join(PROJECT_ROOT, 'output') };
  }
}

/**
 * Read the training progress manifest from output dir.
 */
function readProgressManifest(outputDir: string): Record<string, any> | null {
  const manifestPath = path.join(outputDir, 'job_manifest.json');
  if (!fs.existsSync(manifestPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  } catch {
    return null;
  }
}

/**
 * Sync job store status from the output manifest.
 */
function syncJobStatus(jobId: string, outputDir: string): void {
  const manifest = readProgressManifest(outputDir);
  if (!manifest) return;

  const store = getJobStore();
  const job = store.getJob(jobId);
  if (!job) return;

  // Update status from manifest
  if (manifest.status === 'completed' && job.status === 'running') {
    store.updateStatus(jobId, 'completed');
  } else if (manifest.status === 'failed' && job.status === 'running') {
    store.updateStatus(jobId, 'failed');
    if (manifest.error) {
      store.updateError(jobId, manifest.error);
    }
  }
}

/**
 * Launch a training job via uv run scripts/train_single.py.
 */
function launchTraining(jobId: string, params: TrainingParams, outputDir: string): Promise<void> {
  return new Promise((resolve) => {
    const trainingParams = {
      ...params,
      output_dir: outputDir,
      job_id: jobId,
    };

    // Write params to a temp file to avoid shell escaping issues on Windows
    const paramsFile = path.join(os.tmpdir(), `train-params-${jobId}.json`);
    fs.writeFileSync(paramsFile, JSON.stringify(trainingParams), 'utf8');

    const cmd = 'uv';
    const args = [
      'run',
      'python',
      path.join(PROJECT_ROOT, 'scripts', 'train_single.py'),
      '--params-json-file',
      paramsFile,
    ];

    const proc = spawn(cmd, args, { shell: true, cwd: PROJECT_ROOT });

    // Store the actual proc reference so cancel can kill it directly
    activeJobs.set(jobId, { params, proc });

    // Also persist PID to disk for recovery after server restart
    if (proc.pid) {
      getJobStore().updatePid(jobId, proc.pid);
    }

    // Capture output for debugging
    proc.stdout.on('data', (data: Buffer) => {
      console.log(`[train:${jobId}] stdout: ${data.toString().trim()}`);
    });

    proc.stderr.on('data', (data: Buffer) => {
      console.error(`[train:${jobId}] stderr: ${data.toString().trim()}`);
    });

    proc.on('close', (code) => {
      syncJobStatus(jobId, outputDir);

      // Clean up temp params file
      try {
        if (fs.existsSync(paramsFile)) {
          fs.unlinkSync(paramsFile);
        }
      } catch {
        // Ignore cleanup errors
      }

      activeJobs.delete(jobId);
      console.log(`[train:${jobId}] process exited with code ${code}`);
      resolve();
    });

    // Poll the output manifest to sync job status while running
    const pollInterval = setInterval(() => {
      syncJobStatus(jobId, outputDir);
    }, 5000);

    // Stop polling when process exits
    proc.on('exit', () => {
      clearInterval(pollInterval);
    });
  });
}

/**
 * POST /api/train
 *
 * Validates training parameters and launches a single training job.
 *
 * Body: TrainingParams (validated by trainingSchema)
 * Returns: { jobId, status: "started" }
 */
export async function POST(request: Request) {
  try {
    // Check if a job is already running
    if (activeJobs.size > 0) {
      const [currentJobId] = activeJobs.keys();
      return NextResponse.json(
        {
          error: 'A training job is already running',
          currentJobId,
          hint: 'Wait for the current job to complete, or use Matrix mode to train multiple configurations at once.',
        },
        { status: 409 }
      );
    }

    // Parse and validate request body
    const body = await request.json();
    const result = trainingSchema.safeParse(body);

    if (!result.success) {
      const errors = (result.error.issues || []).map((e) => ({
        field: e.path.join('.'),
        message: e.message,
      }));

      return NextResponse.json(
        {
          error: 'Validation failed',
          details: errors,
        },
        { status: 400 }
      );
    }

    const params = result.data;

    // Load config to get user's output directory
    const config = loadConfig();
    const userOutputDir = config.outputDir;

    // Create job in store (gets its own ID)
    const store = getJobStore();
    const jobId = store.createJob(params);

    // Build output directory: <userOutputDir>/<jobId>/
    const jobOutputDir = path.join(userOutputDir, jobId);

    // Check if a folder with the same loraName already exists in the user's output dir
    if (fs.existsSync(userOutputDir)) {
      const existing = fs.readdirSync(userOutputDir);
      if (existing.includes(params.loraName)) {
        store.deleteJob(jobId);
        return NextResponse.json(
          {
            error: `A training with name "${params.loraName}" already exists in the output folder. Please choose a different name.`,
          },
          { status: 409 }
        );
      }
    }

    // Update job record with the actual output directory
    const job = store.getJob(jobId);
    if (job) {
      job.params.outputDir = jobOutputDir;
    }
    store.updateStatus(jobId, 'running');

    // Create output directory
    fs.mkdirSync(jobOutputDir, { recursive: true });

    // Create zip of training data (best-effort, don't block training)
    try {
      await createTrainingZip(params.trainingImages, jobOutputDir);
    } catch (err: any) {
      console.warn(`[train:${jobId}] zip creation skipped: ${err.message}`);
    }

    // Track the active job (proc reference updated inside launchTraining)
    activeJobs.set(jobId, { params, proc: null });

    // Launch training asynchronously
    launchTraining(jobId, params, jobOutputDir).catch((err) => {
      console.error(`[train:${jobId}] launch error:`, err);
      store.updateStatus(jobId, 'failed');
      store.updateError(jobId, err.message || 'Unknown launch error');
      activeJobs.delete(jobId);
    });

    return NextResponse.json({
      jobId,
      status: 'started',
      message: `Training job ${jobId} has been started`,
      outputDir: jobOutputDir,
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        error: error.message || 'Failed to start training',
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/train/[jobId]/cancel
 * Handles cancellation via a separate route (see api/jobs/[jobId]/cancel).
 * This function is exported for internal use.
 */
export function cancelJob(jobId: string): boolean {
  const activeJob = activeJobs.get(jobId);
  if (!activeJob) {
    return false;
  }

  // Write cancel signal file for Python script
  const cancelPath = path.join(PROJECT_ROOT, 'jobs', `${jobId}.cancel`);
  fs.writeFileSync(cancelPath, new Date().toISOString());

  // Kill the child process if we have a reference
  if (activeJob.proc) {
    try {
      activeJob.proc.kill('SIGTERM');
    } catch {
      // Ignore kill errors
    }
  }

  const store = getJobStore();
  store.updateStatus(jobId, 'failed');
  store.updateError(jobId, 'Cancelled by user');
  activeJobs.delete(jobId);

  return true;
}
