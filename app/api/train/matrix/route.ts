import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { getJobStore } from '../../../lib/job-store';
import { createTrainingZip } from '../../../lib/training-zip';

const PROJECT_ROOT = path.resolve(process.cwd());

/**
 * Parse a comma-separated parameter range string into an array.
 * This is a simple client-side version of the Python parser.
 */
function parseParamRange(valueStr: string): (number | string)[] {
  if (!valueStr || !valueStr.trim()) return [];

  return valueStr.split(',').map((p) => {
    const trimmed = p.trim();
    if (!trimmed) return null;

    // Preserve % markers
    if (trimmed.endsWith('%')) return trimmed;

    // Try number
    const num = Number(trimmed);
    if (!isNaN(num) && isFinite(num)) return num;

    return trimmed;
  }).filter((v): v is string | number => v !== null);
}

/**
 * POST /api/train/matrix
 *
 * Validates matrix parameters and launches the matrix trainer script.
 *
 * Body: {
 *   paramRanges: { param_name: "val1,val2,val3", ... },
 *   baseParams: { trainingImages, loraName, epochs, ... }
 * }
 * Returns: { jobId, permutationCount, status: "started" }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { paramRanges, baseParams = {} } = body as {
      paramRanges?: Record<string, string>;
      baseParams?: Record<string, any>;
    };

    if (!paramRanges || typeof paramRanges !== 'object') {
      return NextResponse.json(
        { error: 'paramRanges is required and must be an object' },
        { status: 400 }
      );
    }

    // Parse and validate parameter ranges
    const parsedRanges: Record<string, (number | string)[]> = {};
    let permutationCount = 1;

    for (const [key, valueStr] of Object.entries(paramRanges)) {
      if (typeof valueStr !== 'string') {
        return NextResponse.json(
          { error: `Parameter range for ${key} must be a comma-separated string` },
          { status: 400 }
        );
      }

      const values = parseParamRange(valueStr);
      if (values.length === 0) {
        return NextResponse.json(
          { error: `Parameter range for ${key} is empty` },
          { status: 400 }
        );
      }

      parsedRanges[key] = values;
      permutationCount *= values.length;
    }

    if (permutationCount === 0 || Object.keys(parsedRanges).length === 0) {
      return NextResponse.json(
        { error: 'No permutations would be generated' },
        { status: 400 }
      );
    }

    // Create job
    const store = getJobStore();
    const jobId = store.createJob({
      type: 'matrix',
      paramRanges: parsedRanges,
      baseParams,
      permutationCount,
    });

    // Create zip of training data (best-effort)
    const outputDir = path.join(PROJECT_ROOT, 'output', jobId);
    const trainingImagesPath = (baseParams as any).trainingImages;
    let zipPath: string | null = null;
    if (trainingImagesPath) {
      try {
        zipPath = await createTrainingZip(trainingImagesPath, outputDir);
      } catch (err: any) {
        console.warn(`[matrix:${jobId}] zip creation skipped: ${err.message}`);
      }
    }

    // Write JSON to temp files to avoid shell escaping issues on Windows
    const paramRangesFile = path.join(outputDir, 'param_ranges.json');
    const baseParamsFile = path.join(outputDir, 'base_params.json');
    fs.mkdirSync(outputDir, { recursive: true });
    fs.writeFileSync(paramRangesFile, JSON.stringify(parsedRanges));
    fs.writeFileSync(baseParamsFile, JSON.stringify(baseParams));

    const cmd = 'uv';
    const args = [
      'run',
      'python',
      path.join(PROJECT_ROOT, 'scripts', 'matrix_trainer.py'),
      '--param-ranges-file',
      paramRangesFile,
      '--base-params-file',
      baseParamsFile,
      '--output-dir',
      outputDir,
    ];

    const proc = spawn(cmd, args, {
      shell: true,
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONPATH: PROJECT_ROOT },
    });

    store.updateStatus(jobId, 'running');

    proc.stdout.on('data', (data: Buffer) => {
      console.log(`[matrix:${jobId}] stdout: ${data.toString().trim()}`);
    });

    proc.stderr.on('data', (data: Buffer) => {
      console.error(`[matrix:${jobId}] stderr: ${data.toString().trim()}`);
    });

    proc.on('close', (code) => {
      const finalStatus = code === 0 ? 'completed' : 'failed';
      store.updateStatus(jobId, finalStatus);
    });

    return NextResponse.json({
      jobId,
      permutationCount,
      status: 'started',
      message: `Matrix training started with ${permutationCount} permutations`,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to start matrix training' },
      { status: 500 }
    );
  }
}
