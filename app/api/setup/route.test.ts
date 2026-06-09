import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextResponse } from 'next/server';

// --- Mock next/server ---
vi.mock('next/server', () => ({
  NextResponse: {
    json: (data: any, init?: ResponseInit) => ({
      json: async () => data,
      status: init?.status ?? 200,
      headers: new Headers(init?.headers),
    }),
  },
}));

// --- Mock child_process ---
const mockSpawn = vi.fn();
vi.mock('child_process', () => ({
  default: { spawn: mockSpawn },
  spawn: mockSpawn,
}));

// --- Mock fs ---
const mockExistsSync = vi.fn();
const mockStatSync = vi.fn();
const mockMkdirSync = vi.fn();
const mockWriteFileSync = vi.fn();
const mockRmSync = vi.fn();
const mockUnlinkSync = vi.fn();
vi.mock('fs', () => ({
  default: {
    readFileSync: vi.fn(),
    writeFileSync: mockWriteFileSync,
    existsSync: mockExistsSync,
    statSync: mockStatSync,
    mkdirSync: mockMkdirSync,
    rmSync: mockRmSync,
    unlinkSync: mockUnlinkSync,
  },
  existsSync: mockExistsSync,
  statSync: mockStatSync,
  mkdirSync: mockMkdirSync,
  rmSync: mockRmSync,
  unlinkSync: mockUnlinkSync,
}));

// --- Mock path ---
vi.mock('path', () => ({
  default: {
    resolve: (...args: string[]) => args.join('/'),
    join: (...args: string[]) => args.join('/'),
  },
  resolve: (...args: string[]) => args.join('/'),
  join: (...args: string[]) => args.join('/'),
}));

// --- Mock setup-tracker ---
const mockReadSetupStatus = vi.fn();
const mockWriteSetupStatus = vi.fn();
const mockUpdateStep = vi.fn();
const mockResetSetupStatus = vi.fn();
const mockCheckReadiness = vi.fn();
vi.mock('../../lib/setup-tracker', () => ({
  readSetupStatus: mockReadSetupStatus,
  writeSetupStatus: mockWriteSetupStatus,
  updateStep: mockUpdateStep,
  resetSetupStatus: mockResetSetupStatus,
  checkReadiness: mockCheckReadiness,
}));

async function importRoute() {
  // Clear module cache so mocks take effect
  vi.resetModules();
  return await import('./route');
}

function createMockProcess(stdoutData: string, stderrData: string, exitCode: number) {
  return {
    stdout: {
      on: (event: string, cb: (data: Buffer) => void) => {
        if (event === 'data' && stdoutData) cb(Buffer.from(stdoutData));
      },
      removeAllListeners: () => {},
    },
    stderr: {
      on: (event: string, cb: (data: Buffer) => void) => {
        if (event === 'data' && stderrData) cb(Buffer.from(stderrData));
      },
      removeAllListeners: () => {},
    },
    on: (event: string, cb: (code: number | null) => void) => {
      if (event === 'close') cb(exitCode);
    },
  };
}

describe('/api/setup', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('GET', () => {
    it('returns readiness status from checkReadiness + setup status', async () => {
      mockCheckReadiness.mockReturnValue({ venvReady: true, sdScriptsReady: true });
      mockReadSetupStatus.mockReturnValue({
        status: 'idle',
        currentStep: null,
        steps: {},
        updatedAt: new Date().toISOString(),
      });

      const route = await importRoute();
      const response = await route.GET();
      const body = await response.json();

      expect(body.venvReady).toBe(true);
      expect(body.sdScriptsReady).toBe(true);
      expect(body.setup).toBeDefined();
    });

    it('returns venvReady: false when venv does not exist', async () => {
      mockCheckReadiness.mockReturnValue({ venvReady: false, sdScriptsReady: false });
      mockReadSetupStatus.mockReturnValue({
        status: 'idle',
        currentStep: null,
        steps: {},
        updatedAt: new Date().toISOString(),
      });

      const route = await importRoute();
      const response = await route.GET();
      const body = await response.json();

      expect(body.venvReady).toBe(false);
      expect(body.sdScriptsReady).toBe(false);
    });
  });

  describe('POST', () => {
    it('returns immediately with setup started message', async () => {
      mockReadSetupStatus.mockReturnValue({
        status: 'idle',
        currentStep: null,
        steps: {},
        updatedAt: new Date().toISOString(),
      });

      const route = await importRoute();
      const response = await route.POST();
      const body = await response.json();

      expect(body.message).toBe('Setup started');
    });

    it('rejects if setup is already running', async () => {
      mockReadSetupStatus.mockReturnValue({
        status: 'running',
        currentStep: 'uv-sync',
        steps: {},
        updatedAt: new Date().toISOString(),
      });

      const route = await importRoute();
      const response = await route.POST();
      const body = await response.json();

      expect(body.message).toBe('Setup already in progress');
    });
  });
});

describe('parseGpuInfo', () => {
  async function getParseGpuInfo() {
    const route = await importRoute();
    return route.parseGpuInfo;
  }

  it('returns cu130 for RTX 50 series from query output', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    // nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    const result = parseGpuInfo('NVIDIA GeForce RTX 5090, 12.0, 596.49');
    expect(result).toEqual({
      cuda: 'cu130',
      series: 'blackwell',
      gpuName: 'NVIDIA GeForce RTX 5090',
      cudaVersion: '596.49',
      computeCapability: '12.0',
    });
  });

  it('returns cu128 for RTX 40 series from query output', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo('NVIDIA GeForce RTX 4090, 8.9, 535.00');
    expect(result).toEqual({
      cuda: 'cu128',
      series: 'ada',
      gpuName: 'NVIDIA GeForce RTX 4090',
      cudaVersion: '535.00',
      computeCapability: '8.9',
    });
  });

  it('returns cu128 for RTX 30 series from query output', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo('NVIDIA GeForce RTX 3080, 8.6, 535.00');
    expect(result).toEqual({
      cuda: 'cu128',
      series: 'ampere',
      gpuName: 'NVIDIA GeForce RTX 3080',
      cudaVersion: '535.00',
      computeCapability: '8.6',
    });
  });

  it('returns null for unsupported GPU', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo('NVIDIA GeForce GTX 1080, 6.1, 535.00');
    expect(result).toBeNull();
  });

  it('returns null for empty input', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo('');
    expect(result).toBeNull();
  });

  it('returns null for malformed output (single column)', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo('NVIDIA GeForce RTX 4090');
    expect(result).toBeNull();
  });
});
