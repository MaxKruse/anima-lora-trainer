import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getJobStore } from '../../../lib/job-store';

const PROJECT_ROOT = path.resolve(process.cwd());

/**
 * Resolve the output directory for a job.
 */
function resolveOutputDir(job: any): string | null {
  if (job?.params?.outputDir) {
    return job.params.outputDir;
  }
  // Legacy fallback
  return path.join(PROJECT_ROOT, 'output', job.id);
}

/**
 * DELETE /api/jobs/[jobId]
 *
 * Deletes a job record and its output directory.
 * Only allows deletion of non-running jobs.
 */
export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  try {
    const { jobId } = await params;
    const store = getJobStore();
    const job = store.getJob(jobId);

    if (!job) {
      return NextResponse.json(
        { error: `Job ${jobId} not found` },
        { status: 404 }
      );
    }

    if (job.status === 'running') {
      return NextResponse.json(
        { error: 'Cannot delete a running job' },
        { status: 409 }
      );
    }

    // Delete output directory
    const outputDir = resolveOutputDir(job);
    if (outputDir) {
      try {
        if (fs.existsSync(outputDir)) {
          fs.rmSync(outputDir, { recursive: true, force: true });
        }
      } catch {
        // Ignore output dir deletion errors
      }
    }

    // Clean up cancel signal file if it exists
    try {
      const cancelPath = path.join(PROJECT_ROOT, 'jobs', `${jobId}.cancel`);
      if (fs.existsSync(cancelPath)) {
        fs.unlinkSync(cancelPath);
      }
    } catch {
      // Ignore cleanup errors
    }

    // Delete job record
    store.deleteJob(jobId);

    return NextResponse.json({ success: true, jobId });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to delete job' },
      { status: 500 }
    );
  }
}
