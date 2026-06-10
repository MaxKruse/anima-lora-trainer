import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

const PROJECT_ROOT = path.resolve(process.cwd());
const CONFIG_DIR = path.join(PROJECT_ROOT, '.config');
const CONFIG_FILE = path.join(CONFIG_DIR, 'app-config.json');

function loadConfig(): { outputDir: string } {
  try {
    if (!fs.existsSync(CONFIG_FILE)) {
      return { outputDir: path.join(PROJECT_ROOT, 'output') };
    }
    const raw = fs.readFileSync(CONFIG_FILE, 'utf8');
    const saved = JSON.parse(raw);
    return {
      outputDir: saved.outputDir || path.join(PROJECT_ROOT, 'output'),
    };
  } catch {
    return { outputDir: path.join(PROJECT_ROOT, 'output') };
  }
}

function getOutputDir(): string {
  return loadConfig().outputDir;
}

/**
 * Track which runs are currently being evaluated.
 * Map from runId to process reference.
 */
const activeEvaluations = new Map<string, { proc: any; startedAt: string }>();

/**
 * Reset evaluation state. Used for testing.
 */
export function __resetEvalState(): void {
  for (const entry of activeEvaluations.values()) {
    entry.proc.kill();
  }
  activeEvaluations.clear();
}

/**
 * Resolve the run directory for a given run ID.
 */
function resolveRunDir(runId: string): string {
  return path.join(getOutputDir(), runId);
}

/**
 * Launch the matrix evaluator script for a run.
 */
function launchEvaluation(runId: string): void {
  const runDir = resolveRunDir(runId);

  const cmd = 'uv';
  const args = [
    'run',
    'python',
    path.join(PROJECT_ROOT, 'scripts', 'matrix_evaluator.py'),
    '--run-dir', runDir,
    '--diffusion-model', path.join(PROJECT_ROOT, 'models', 'anima', 'diffusion.safetensors'),
    '--vae-model', path.join(PROJECT_ROOT, 'models', 'anima', 'vae.safetensors'),
    '--llm-model', path.join(PROJECT_ROOT, 'models', 'anima', 'text_encoder', 'model.safetensors'),
    '--prompt', 'cat dog bird sunset landscape',
    '--seed', '42',
  ];

  const proc = spawn(cmd, args, { shell: true, cwd: PROJECT_ROOT });

  activeEvaluations.set(runId, { proc, startedAt: new Date().toISOString() });

  proc.stdout.on('data', (data: Buffer) => {
    console.log(`[eval:${runId}] stdout: ${data.toString().trim()}`);
  });

  proc.stderr.on('data', (data: Buffer) => {
    console.error(`[eval:${runId}] stderr: ${data.toString().trim()}`);
  });

  proc.on('close', (code) => {
    console.log(`[eval:${runId}] exited with code ${code}`);
    activeEvaluations.delete(runId);
  });
}

/**
 * POST /api/evaluate
 *
 * Start evaluation for a completed training run.
 *
 * Body: { runId: string }
 * Returns: { runId, status: "started" }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { runId } = body;

    if (!runId) {
      return NextResponse.json(
        { error: 'runId is required' },
        { status: 400 }
      );
    }

    // Check if the run directory exists
    const runDir = resolveRunDir(runId);
    if (!fs.existsSync(runDir)) {
      return NextResponse.json(
        { error: `Run ${runId} not found` },
        { status: 404 }
      );
    }

    // Check if evaluation is already running for this run
    if (activeEvaluations.has(runId)) {
      return NextResponse.json(
        {
          error: `Evaluation already running for ${runId}`,
          startedAt: activeEvaluations.get(runId)?.startedAt,
        },
        { status: 409 }
      );
    }

    // Launch evaluation asynchronously
    launchEvaluation(runId);

    return NextResponse.json({
      runId,
      status: 'started',
      message: `Evaluation for ${runId} has been started`,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to start evaluation' },
      { status: 500 }
    );
  }
}

/**
 * GET /api/evaluate?runId=X
 *
 * Return evaluation results for a completed run.
 *
 * Query: runId (required)
 * Returns: evaluation.json contents or 404
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const runId = searchParams.get('runId');

    if (!runId) {
      return NextResponse.json(
        { error: 'runId query parameter is required' },
        { status: 400 }
      );
    }

    const evalPath = path.join(resolveRunDir(runId), 'evaluation.json');

    if (!fs.existsSync(evalPath)) {
      return NextResponse.json(
        { error: `Evaluation results not found for ${runId}` },
        { status: 404 }
      );
    }

    const content = fs.readFileSync(evalPath, 'utf-8');
    const data = JSON.parse(content);

    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch evaluation results' },
      { status: 500 }
    );
  }
}
