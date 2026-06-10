import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getJobStore } from '../../../lib/job-store';

const PROJECT_ROOT = path.resolve(process.cwd());

/**
 * Resolve the output directory for a job.
 * Uses the job record's outputDir if available, falls back to legacy path.
 */
function resolveOutputDir(jobId: string): string | null {
  const store = getJobStore();
  const job = store.getJob(jobId);

  // New: use outputDir from job params
  if (job?.params?.outputDir) {
    return job.params.outputDir;
  }

  // Legacy fallback: output/<jobId>
  return path.join(PROJECT_ROOT, 'output', jobId);
}

/**
 * GET /api/logs/[jobId]
 *
 * Returns the training log file for a job.
 * Supports ?tail=N to get only the last N lines.
 * Supports ?follow (SSE) for real-time log streaming.
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  try {
    const { jobId } = await params;
    const { searchParams } = new URL(request.url);
    const tail = searchParams.get('tail');
    const follow = searchParams.has('follow');

    const outputDir = resolveOutputDir(jobId);
    if (!outputDir) {
      return NextResponse.json(
        { error: 'Job not found', jobId },
        { status: 404 }
      );
    }

    const logPath = path.join(outputDir, 'training.log');

    if (!fs.existsSync(logPath)) {
      return NextResponse.json(
        { error: 'Log file not found', jobId },
        { status: 404 }
      );
    }

    if (follow) {
      return serveLogFollow(logPath, jobId);
    }

    const content = fs.readFileSync(logPath, 'utf8');
    const lines = content.split('\n');

    if (tail) {
      const n = parseInt(tail, 10);
      if (!isNaN(n) && n > 0) {
        return NextResponse.json({
          jobId,
          lines: lines.slice(-n),
          totalLines: lines.length,
        });
      }
    }

    return NextResponse.json({
      jobId,
      lines,
      totalLines: lines.length,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to read log' },
      { status: 500 }
    );
  }
}

/**
 * Serve log file as Server-Sent Events (SSE) for real-time streaming.
 */
function serveLogFollow(logPath: string, jobId: string) {
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      let position = 0;
      let interval: ReturnType<typeof setInterval> | null = null;

      function read() {
        try {
          const stats = fs.statSync(logPath);
          const size = stats.size;

          if (size > position) {
            const fd = fs.openSync(logPath, 'r');
            const buffer = Buffer.alloc(size - position);
            fs.readSync(fd, buffer, 0, buffer.length, position);
            fs.closeSync(fd);

            const chunk = buffer.toString('utf8');
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ jobId, chunk })}\n\n`)
            );
            position = size;
          }
        } catch {
          // File may not exist yet or be inaccessible
        }
      }

      read();
      interval = setInterval(read, 1000);

      return () => {
        if (interval) clearInterval(interval);
      };
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}
