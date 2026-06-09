import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { trainingSchema, type TrainingParams } from '../../lib/training-schema';
import { createTrainingZip } from '../../lib/training-zip';

const PROJECT_ROOT = path.resolve(process.cwd());
const JOBS_DIR = path.join(PROJECT_ROOT, 'jobs');

/**
 * In-memory state to track the currently running job.
 */
let currentJob: { jobId: string; params: TrainingParams } | null = null;

/**
 * Reset current job state. Used for testing.
 */
export function __resetJobState(): void {
  currentJob = null;
}

/**
 * Generate a unique job ID using timestamp + random suffix.
 */
function generateJobId(): string {
  const timestamp = Date.now();
  const randomSuffix = Math.random().toString(36).substring(2, 8);
  return `job-${timestamp}-${randomSuffix}`;
}

/**
 * Ensure the jobs directory exists.
 */
function ensureJobsDir(): void {
  if (!fs.existsSync(JOBS_DIR)) {
    fs.mkdirSync(JOBS_DIR, { recursive: true });
  }
}

/**
 * Write a job manifest file.
 */
function writeJobManifest(jobId: string, data: Record<string, any>): void {
  ensureJobsDir();
  const manifestPath = path.join(JOBS_DIR, `${jobId}.json`);
  fs.writeFileSync(manifestPath, JSON.stringify(data, null, 2));
}

/**
 * Launch a training job via uv run scripts/train_single.py.
 */
async function launchTraining(jobId: string, params: TrainingParams): Promise<void> {
  const outputDir = path.join(PROJECT_ROOT, 'output', jobId);

  // Create zip of training data (best-effort, don't block training)
  let zipPath: string | null = null;
  try {
    zipPath = await createTrainingZip(params.trainingImages, outputDir);
  } catch (err: any) {
    console.warn(`[train:${jobId}] zip creation skipped: ${err.message}`);
  }

  const trainingParams = {
    ...params,
    output_dir: outputDir,
  };

  const cmd = 'uv';
  const args = [
    'run',
    'python',
    path.join(PROJECT_ROOT, 'scripts', 'train_single.py'),
    '--params-json',
    JSON.stringify(trainingParams),
  ];

  const proc = spawn(cmd, args, { shell: true, cwd: PROJECT_ROOT });

  // Write initial job manifest
  writeJobManifest(jobId, {
    jobId,
    status: 'running',
    params: trainingParams,
    outputDir,
    zipPath: zipPath || undefined,
    startedAt: new Date().toISOString(),
    pid: proc.pid,
  });

  // Capture output for debugging
  proc.stdout.on('data', (data: Buffer) => {
    console.log(`[train:${jobId}] stdout: ${data.toString().trim()}`);
  });

  proc.stderr.on('data', (data: Buffer) => {
    console.error(`[train:${jobId}] stderr: ${data.toString().trim()}`);
  });

  proc.on('close', (code) => {
    const finalStatus = code === 0 ? 'completed' : 'failed';
    writeJobManifest(jobId, {
      jobId,
      status: finalStatus,
      exitCode: code,
      completedAt: new Date().toISOString(),
    });
    currentJob = null;
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
    // Check if another job is running
    if (currentJob) {
      return NextResponse.json(
        {
          error: 'A training job is already running',
          currentJobId: currentJob.jobId,
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
    const jobId = generateJobId();

    // Track the job in memory
    currentJob = { jobId, params };

    // Launch training asynchronously
    launchTraining(jobId, params).catch((err) => {
      console.error(`[train:${jobId}] launch error:`, err);
      currentJob = null;
    });

    return NextResponse.json({
      jobId,
      status: 'started',
      message: `Training job ${jobId} has been started`,
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
