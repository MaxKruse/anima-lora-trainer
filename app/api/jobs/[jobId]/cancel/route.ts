import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { getJobStore } from '../../../../lib/job-store';
import { cancelJob } from '../../../train/route';

const PROJECT_ROOT = path.resolve(process.cwd());

/**
 * POST /api/jobs/[jobId]/cancel
 *
 * Cancels a running training job by writing a cancel signal file
 * and updating the job status.
 */
export async function POST(
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

    if (job.status !== 'running') {
      return NextResponse.json(
        { error: `Job is not running (current status: ${job.status})` },
        { status: 409 }
      );
    }

    // Use the shared cancelJob function: writes signal file AND kills proc directly
    const cancelled = cancelJob(jobId);

    return NextResponse.json({
      success: true,
      jobId,
      message: cancelled ? 'Cancel signal sent. Training will stop immediately.' : 'Cancel signal sent. Training will stop shortly.',
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to cancel job' },
      { status: 500 }
    );
  }
}
