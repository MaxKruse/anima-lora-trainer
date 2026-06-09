/**
 * Tracks setup progress in .config/setup-status.json so the UI can poll for real-time updates.
 */

import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.resolve(process.cwd());
const CONFIG_DIR = path.join(PROJECT_ROOT, '.config');
const SETUP_STATUS_FILE = path.join(CONFIG_DIR, 'setup-status.json');

export type SetupStep =
  | 'detect-gpu'
  | 'generate-pyproject'
  | 'clean-venv'
  | 'uv-sync'
  | 'verify-pytorch-cuda'
  | 'clone-sd-scripts'
  | 'done';

export interface SetupStatus {
  status: 'idle' | 'running' | 'success' | 'error';
  currentStep: SetupStep | null;
  steps: Record<
    SetupStep,
    { status: 'pending' | 'running' | 'done' | 'error'; output?: string }
  >;
  error?: string;
  gpu?: string;
  series?: string;
  cuda?: string;
  computeCapability?: string;
  pytorchCudaVersion?: string;
  updatedAt: string;
}

const ALL_STEPS: SetupStep[] = [
  'detect-gpu',
  'generate-pyproject',
  'clean-venv',
  'uv-sync',
  'verify-pytorch-cuda',
  'clone-sd-scripts',
  'done',
];

function emptyStatus(): SetupStatus {
  const steps: SetupStatus['steps'] = {} as any;
  for (const step of ALL_STEPS) {
    steps[step] = { status: 'pending' };
  }
  return {
    status: 'idle',
    currentStep: null,
    steps,
    updatedAt: new Date().toISOString(),
  };
}

export function readSetupStatus(): SetupStatus {
  try {
    if (!fs.existsSync(SETUP_STATUS_FILE)) return emptyStatus();
    const raw = fs.readFileSync(SETUP_STATUS_FILE, 'utf-8');
    const parsed = JSON.parse(raw);
    // Merge with defaults for any missing steps
    const base = emptyStatus();
    return { ...base, ...parsed, steps: { ...base.steps, ...parsed.steps } };
  } catch {
    return emptyStatus();
  }
}

export function writeSetupStatus(updates: Partial<SetupStatus>): void {
  const current = readSetupStatus();
  const updated = { ...current, ...updates, updatedAt: new Date().toISOString() };
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  fs.writeFileSync(SETUP_STATUS_FILE, JSON.stringify(updated, null, 2));
}

export function updateStep(
  step: SetupStep,
  patch: { status: 'running' | 'done' | 'error'; output?: string }
): void {
  writeSetupStatus({
    currentStep: patch.status === 'running' ? step : null,
    steps: {
      ...readSetupStatus().steps,
      [step]: {
        ...(readSetupStatus().steps[step] ?? { status: 'pending' }),
        ...patch,
      },
    },
  });
}

/** Reset status to idle so a fresh setup can be started. */
export function resetSetupStatus(): void {
  fs.writeFileSync(SETUP_STATUS_FILE, JSON.stringify(emptyStatus(), null, 2));
}

/** Check if all prerequisites are met (venv + sd-scripts). */
export function checkReadiness(): { venvReady: boolean; sdScriptsReady: boolean } {
  const venvPath = path.join(PROJECT_ROOT, '.venv');
  const sdScriptsPath = path.join(PROJECT_ROOT, 'sd-scripts');

  return {
    venvReady: fs.existsSync(venvPath) && fs.statSync(venvPath).isDirectory(),
    sdScriptsReady: fs.existsSync(sdScriptsPath) && fs.statSync(sdScriptsPath).isDirectory(),
  };
}
