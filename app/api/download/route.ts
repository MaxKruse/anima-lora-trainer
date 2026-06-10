import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getJobStore } from '../../lib/job-store';

const PROJECT_ROOT = path.resolve(process.cwd());
const DEFAULT_OUTPUT_DIR = path.join(PROJECT_ROOT, 'output');

/**
 * Recursively search for a file in a directory.
 */
function findFileInDir(dir: string, fileName: string): string | null {
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const found = findFileInDir(fullPath, fileName);
        if (found) return found;
      } else if (entry.name === fileName) {
        return fullPath;
      }
    }
  } catch {
    // Ignore read errors
  }
  return null;
}

/**
 * Resolve the output directory for a given runId/jobId.
 *
 * Tries:
 * 1. Job store — looks up params.outputDir for the job
 * 2. Default output/<runId>/ — for matrix runs or legacy jobs
 */
function resolveOutputDir(runId: string): string | null {
  // Try job store first (single training jobs store outputDir in params)
  const store = getJobStore();
  const job = store.getJob(runId);
  if (job?.params?.outputDir) {
    return job.params.outputDir;
  }

  // Fall back to default output directory
  const defaultDir = path.join(DEFAULT_OUTPUT_DIR, runId);
  if (fs.existsSync(defaultDir)) {
    return defaultDir;
  }

  // Also try scanning all job outputDirs for a match
  for (const j of store.listJobs()) {
    if (j.params?.outputDir && fs.existsSync(j.params.outputDir)) {
      // Check if this directory contains the file we're looking for
      // (covered by caller, but at least return a valid dir)
      return j.params.outputDir;
    }
  }

  return null;
}

/**
 * GET /api/download?runId=X&file=Y
 *
 * Serve a file from a training run's output directory.
 * Prevents path traversal attacks by using path.basename().
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const runId = searchParams.get('runId');
    const file = searchParams.get('file');

    if (!runId || !file) {
      return NextResponse.json(
        { error: 'runId and file are required' },
        { status: 400 }
      );
    }

    // Prevent path traversal
    const safeFile = path.basename(file);
    const safeRunId = path.basename(runId);

    // Resolve the actual output directory
    const runDir = resolveOutputDir(safeRunId);
    if (!runDir || !fs.existsSync(runDir)) {
      return NextResponse.json(
        { error: 'Run directory not found' },
        { status: 404 }
      );
    }

    // Search for the file in the run directory (may be in subdirectories)
    const filePath = findFileInDir(runDir, safeFile);
    if (!filePath || !fs.existsSync(filePath)) {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }

    const content = fs.readFileSync(filePath);
    return new Response(content, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': `attachment; filename="${safeFile}"`,
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to serve file' },
      { status: 500 }
    );
  }
}
