import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

const PROJECT_ROOT = path.resolve(process.cwd());

/**
 * Run a command and capture stdout/stderr.
 */
export function runCommand(
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

/**
 * Detect GPU from nvidia-smi output.
 */
export function parseGpuInfo(nvidiaOutput: string): {
  cuda: string;
  series: string;
  gpuName: string;
} | null {
  if (!nvidiaOutput || !nvidiaOutput.trim()) return null;

  // Extract GPU name
  let gpuName = '';
  for (const line of nvidiaOutput.split('\n')) {
    if (line.includes('GPU Name') && line.includes(':')) {
      gpuName = line.split(':')[1].trim();
      break;
    }
  }
  if (!gpuName) gpuName = nvidiaOutput.trim();

  if (gpuName.includes('RTX 50')) {
    return { cuda: 'cu130', series: 'blackwell', gpuName };
  } else if (gpuName.includes('RTX 40')) {
    return { cuda: 'cu128', series: 'ada', gpuName };
  } else if (gpuName.includes('RTX 30')) {
    return { cuda: 'cu128', series: 'ampere', gpuName };
  }

  return null;
}

/**
 * Generate pyproject.toml with the given CUDA version.
 */
export async function generatePyproject(cuda: string): Promise<string> {
  const tomlPath = path.join(PROJECT_ROOT, 'pyproject.toml');
  const result = await runCommand('uv', [
    'run', 'python',
    path.join(PROJECT_ROOT, 'scripts', 'setup_env.py'),
    '--generate',
    tomlPath,
    cuda,
  ]);

  if (!result.success) {
    throw new Error(`Failed to generate pyproject.toml: ${result.stderr}`);
  }

  return tomlPath;
}

/**
 * POST /api/setup
 *
 * Runs GPU detection via nvidia-smi and generates pyproject.toml.
 */
export async function POST() {
  try {
    // Step 1: Run nvidia-smi
    const nvidiaResult = await runCommand('nvidia-smi');
    if (!nvidiaResult.success) {
      return NextResponse.json(
        { error: nvidiaResult.stderr || 'nvidia-smi failed', status: 'error' },
        { status: 500 }
      );
    }

    // Step 2: Parse GPU info
    const gpuInfo = parseGpuInfo(nvidiaResult.stdout);
    if (!gpuInfo) {
      return NextResponse.json(
        { error: 'Unsupported GPU detected', status: 'error' },
        { status: 500 }
      );
    }

    // Step 3: Generate pyproject.toml
    const pyprojectPath = await generatePyproject(gpuInfo.cuda);

    return NextResponse.json({
      gpu: gpuInfo.gpuName,
      series: gpuInfo.series,
      cuda: gpuInfo.cuda,
      pyprojectPath,
      status: 'ok',
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Setup failed', status: 'error' },
      { status: 500 }
    );
  }
}
