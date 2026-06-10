import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getJobStore } from '../../../lib/job-store';

const PROJECT_ROOT = path.resolve(process.cwd());

interface ProgressData {
  status: string;
  current_epoch?: number;
  total_epochs?: number;
  current_step?: number;
  total_steps?: number;
  avg_loss?: number;
  error?: string;
  exit_code?: number;
}

/**
 * Resolve the output directory for a job.
 */
function resolveOutputDir(jobId: string): string | null {
  const store = getJobStore();
  const job = store.getJob(jobId);

  if (job?.params?.outputDir) {
    return job.params.outputDir;
  }

  // Legacy fallback
  return path.join(PROJECT_ROOT, 'output', jobId);
}

/**
 * GET /api/progress/[jobId]
 *
 * Returns parsed training progress from the job manifest.
 */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  try {
    const { jobId } = await params;

    const outputDir = resolveOutputDir(jobId);
    if (!outputDir) {
      return NextResponse.json(
        { error: 'Job not found', jobId },
        { status: 404 }
      );
    }

    const manifestPath = path.join(outputDir, 'job_manifest.json');

    if (!fs.existsSync(manifestPath)) {
      return NextResponse.json(
        { error: 'Job manifest not found', jobId },
        { status: 404 }
      );
    }

    const content = fs.readFileSync(manifestPath, 'utf8');
    const manifest = JSON.parse(content) as ProgressData;

    // Scan for output .safetensors files when job is completed
    let outputFiles: string[] = [];
    if (manifest.status === 'completed') {
      try {
        outputFiles = fs.readdirSync(outputDir)
          .filter((f) => f.endsWith('.safetensors'))
          .map((f) => f);
      } catch {
        // Ignore scan errors
      }
    }

    return NextResponse.json({
      jobId,
      status: manifest.status || 'unknown',
      currentEpoch: manifest.current_epoch ?? 0,
      totalEpochs: manifest.total_epochs ?? 0,
      currentStep: manifest.current_step ?? 0,
      totalSteps: manifest.total_steps ?? 0,
      avgLoss: manifest.avg_loss ?? null,
      error: manifest.error ?? null,
      exitCode: manifest.exit_code ?? null,
      outputFiles,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to read progress' },
      { status: 500 }
    );
  }
}
