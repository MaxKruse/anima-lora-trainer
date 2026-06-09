import { NextResponse } from 'next/server';
import { getJobStore } from '../../lib/job-store';

/**
 * GET /api/jobs
 *
 * Returns a list of all training jobs, or a single job if ?id=X is provided.
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('id');
    const store = getJobStore();

    if (jobId) {
      // Return single job
      const job = store.getJob(jobId);
      if (!job) {
        return NextResponse.json(
          { error: `Job ${jobId} not found` },
          { status: 404 }
        );
      }
      return NextResponse.json(job);
    }

    // Return all jobs
    const jobs = store.listJobs();
    return NextResponse.json({ jobs });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch jobs' },
      { status: 500 }
    );
  }
}
