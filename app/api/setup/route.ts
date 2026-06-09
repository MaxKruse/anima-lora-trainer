import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import {
  type SetupStatus,
  readSetupStatus,
  writeSetupStatus,
  updateStep,
  resetSetupStatus,
  checkReadiness,
} from '../../lib/setup-tracker';

const PROJECT_ROOT = path.resolve(process.cwd());

// --- Command runner ---

function runCommand(
  command: string,
  args: string[] = []
): Promise<{ success: boolean; stdout: string; stderr: string; code: number | null }> {
  return new Promise((resolve) => {
    const proc = spawn(command, args, { shell: true, cwd: PROJECT_ROOT });
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });
    proc.on('close', (code) => {
      resolve({
        success: code === 0,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        code,
      });
    });
  });
}

// --- GPU detection ---

/**
 * Parse structured nvidia-smi --query-gpu CSV output.
 * Expected: "NVIDIA GeForce RTX 5090, 12.0, 13.2"
 * (name, compute_cap, driver_version)
 */
function parseGpuQuery(output: string): {
  gpuName: string;
  computeCapability: string;
  driverVersion: string;
} | null {
  const lines = output.split('\n').map(l => l.trim()).filter(Boolean);
  if (lines.length === 0) return null;

  // Take first GPU line
  const parts = lines[0].split(',').map(s => s.trim());
  if (parts.length < 2) return null;

  return {
    gpuName: parts[0],
    computeCapability: parts[1] || '',
    driverVersion: parts[2] || '',
  };
}

/**
 * Determine CUDA toolkit series from GPU name.
 */
function detectCudaSeries(gpuName: string): { cuda: string; series: string } | null {
  if (gpuName.includes('RTX 50')) return { cuda: 'cu130', series: 'blackwell' };
  if (gpuName.includes('RTX 40')) return { cuda: 'cu128', series: 'ada' };
  if (gpuName.includes('RTX 30')) return { cuda: 'cu128', series: 'ampere' };
  return null;
}

export function parseGpuInfo(queryOutput: string): {
  cuda: string;
  series: string;
  gpuName: string;
  cudaVersion: string | null;
  computeCapability: string | null;
} | null {
  const parsed = parseGpuQuery(queryOutput);
  if (!parsed || !parsed.gpuName) return null;

  const seriesInfo = detectCudaSeries(parsed.gpuName);
  if (!seriesInfo) return null;

  return {
    cuda: seriesInfo.cuda,
    series: seriesInfo.series,
    gpuName: parsed.gpuName,
    cudaVersion: parsed.driverVersion || null,
    computeCapability: parsed.computeCapability || null,
  };
}

// --- Step implementations ---

async function stepDetectGpu(): Promise<{
  gpuName: string;
  series: string;
  cuda: string;
  cudaVersion: string | null;
  computeCapability: string | null;
}> {
  updateStep('detect-gpu', { status: 'running' });

  // Use structured query — no table parsing needed
  const queryResult = await runCommand('nvidia-smi', [
    '--query-gpu=name,compute_cap,driver_version',
    '--format=csv,noheader',
  ]);

  if (!queryResult.success) {
    updateStep('detect-gpu', { status: 'error', output: queryResult.stderr || 'nvidia-smi failed' });
    throw new Error(queryResult.stderr || 'nvidia-smi failed');
  }

  const gpuInfo = parseGpuInfo(queryResult.stdout);
  if (!gpuInfo) {
    updateStep('detect-gpu', { status: 'error', output: 'Unsupported GPU detected' });
    throw new Error('Unsupported GPU detected. Requires RTX 30-series or newer.');
  }

  updateStep('detect-gpu', {
    status: 'done',
    output: `${gpuInfo.gpuName} (${gpuInfo.series}), Driver ${gpuInfo.cudaVersion || 'unknown'}, Compute ${gpuInfo.computeCapability || 'unknown'}`,
  });
  return gpuInfo;
}

async function stepVerifyPytorchCuda(): Promise<string | null> {
  updateStep('verify-pytorch-cuda', { status: 'running', output: 'Checking PyTorch CUDA version...' });

  const result = await runCommand('uv', [
    'run', 'python', '-c',
    'import torch; print(torch.version.cuda)',
  ]);

  if (!result.success) {
    updateStep('verify-pytorch-cuda', { status: 'error', output: result.stderr || 'Failed to check PyTorch CUDA' });
    return null;
  }

  const version = result.stdout.trim();
  updateStep('verify-pytorch-cuda', { status: 'done', output: `PyTorch CUDA ${version}` });
  return version || null;
}

async function stepGeneratePyproject(cuda: string): Promise<void> {
  updateStep('generate-pyproject', { status: 'running' });

  const tomlPath = path.join(PROJECT_ROOT, 'pyproject.toml');
  const result = await runCommand('uv', [
    'run', 'python',
    path.join(PROJECT_ROOT, 'scripts', 'setup_env.py'),
    '--generate',
    tomlPath,
    cuda,
  ]);

  if (!result.success) {
    updateStep('generate-pyproject', { status: 'error', output: result.stderr });
    throw new Error(`Failed to generate pyproject.toml: ${result.stderr}`);
  }

  updateStep('generate-pyproject', { status: 'done', output: `Wrote ${tomlPath}` });
}

async function stepCleanVenv(): Promise<void> {
  updateStep('clean-venv', { status: 'running' });

  const venvPath = path.join(PROJECT_ROOT, '.venv');
  const lockPath = path.join(PROJECT_ROOT, 'uv.lock');

  if (fs.existsSync(venvPath)) {
    fs.rmSync(venvPath, { recursive: true, force: true });
  }
  if (fs.existsSync(lockPath)) {
    fs.unlinkSync(lockPath);
  }

  updateStep('clean-venv', { status: 'done', output: 'Removed old .venv and uv.lock' });
}

async function stepUvSync(): Promise<void> {
  updateStep('uv-sync', { status: 'running', output: 'Installing dependencies...' });

  const result = await runCommand('uv', ['sync']);

  if (!result.success) {
    // Truncate long output for status storage
    const errorOutput = (result.stderr || result.stdout || 'uv sync failed').slice(-500);
    updateStep('uv-sync', { status: 'error', output: errorOutput });
    throw new Error(`uv sync failed: ${errorOutput}`);
  }

  const lastLines = result.stdout.split('\n').slice(-3).join('\n');
  updateStep('uv-sync', { status: 'done', output: lastLines || 'Dependencies installed successfully' });
}

async function stepCloneSdScripts(): Promise<void> {
  updateStep('clone-sd-scripts', { status: 'running', output: 'Cloning kohya-ss/sd-scripts...' });

  const sdScriptsPath = path.join(PROJECT_ROOT, 'sd-scripts');

  // If it already exists and is a git repo, pull instead
  if (fs.existsSync(sdScriptsPath) && fs.existsSync(path.join(sdScriptsPath, '.git'))) {
    const pullResult = await runCommand('git', ['-C', sdScriptsPath, 'pull']);
    if (pullResult.success) {
      updateStep('clone-sd-scripts', { status: 'done', output: 'Updated sd-scripts via git pull' });
      return;
    }
    // Pull failed — remove and re-clone
    fs.rmSync(sdScriptsPath, { recursive: true, force: true });
  }

  const result = await runCommand('git', [
    'clone',
    'https://github.com/kohya-ss/sd-scripts.git',
    sdScriptsPath,
  ]);

  if (!result.success) {
    updateStep('clone-sd-scripts', { status: 'error', output: result.stderr });
    throw new Error(`Failed to clone sd-scripts: ${result.stderr}`);
  }

  updateStep('clone-sd-scripts', { status: 'done', output: `Cloned to ${sdScriptsPath}` });
}

// --- Full pipeline (runs in background) ---

async function runFullSetup(): Promise<void> {
  // Don't interrupt an already-running setup
  const existing = readSetupStatus();
  if (existing.status === 'running') {
    console.log('[setup] Setup already in progress, skipping');
    return;
  }

  resetSetupStatus();
  writeSetupStatus({ status: 'running' });

  try {
    const gpuInfo = await stepDetectGpu();
    await stepGeneratePyproject(gpuInfo.cuda);
    await stepCleanVenv();
    await stepUvSync();
    const pytorchCudaVersion = await stepVerifyPytorchCuda();
    await stepCloneSdScripts();

    updateStep('done', { status: 'done' });
    writeSetupStatus({
      status: 'success',
      currentStep: null,
      gpu: gpuInfo.gpuName,
      series: gpuInfo.series,
      cuda: gpuInfo.cuda,
      computeCapability: gpuInfo.computeCapability,
      pytorchCudaVersion,
    });
    console.log('[setup] Full setup completed successfully');
  } catch (error: any) {
    writeSetupStatus({ status: 'error', error: error.message });
    console.error('[setup] Setup failed:', error.message);
  }
}

// --- API handlers ---

/**
 * GET /api/setup
 *
 * Returns current setup status (readiness + any in-progress setup job).
 */
export async function GET() {
  const readiness = checkReadiness();
  const setupStatus = readSetupStatus();

  return NextResponse.json({
    venvReady: readiness.venvReady,
    sdScriptsReady: readiness.sdScriptsReady,
    setup: setupStatus,
  });
}

/**
 * POST /api/setup
 *
 * Kicks off the full setup pipeline (GPU detect → pyproject → uv sync → clone sd-scripts).
 * Returns immediately; client should poll GET for progress.
 */
export async function POST() {
  // If already running, tell the client to poll
  const existing = readSetupStatus();
  if (existing.status === 'running') {
    return NextResponse.json({
      message: 'Setup already in progress',
      setup: existing,
    });
  }

  // Kick off background work — don't await, so the response can be sent immediately.
  // The child processes (spawn) keep the Node event loop alive until they finish.
  setImmediate(async () => {
    try {
      await runFullSetup();
    } catch (error: any) {
      console.error('[setup] Unhandled error in setup pipeline:', error);
      writeSetupStatus({ status: 'error', error: error?.message || 'Unknown error' });
    }
  });

  return NextResponse.json({
    message: 'Setup started',
    setup: readSetupStatus(),
  });
}
