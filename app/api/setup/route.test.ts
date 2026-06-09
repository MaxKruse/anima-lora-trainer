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
vi.mock('fs', () => ({
  default: {
    readFileSync: vi.fn(),
    writeFileSync: vi.fn(),
  },
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

async function importRoute() {
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
  });

  it('POST returns gpu info and cuda version when setup succeeds', async () => {
    // First spawn: nvidia-smi (success with RTX 4090 output)
    // Second spawn: uv run python setup_env.py (success)
    mockSpawn
      .mockReturnValueOnce(
        createMockProcess(
          "NVIDIA-SMI 535.00\nGPU Name: NVIDIA GeForce RTX 4090",
          '',
          0
        )
      )
      .mockReturnValueOnce(
        createMockProcess('', '', 0)
      );

    const route = await importRoute();
    const response = await route.POST();
    const body = await response.json();

    expect(body.status).toBe('ok');
    expect(body).toHaveProperty('gpu');
    expect(body).toHaveProperty('cuda');
  });

  it('POST returns 500 when nvidia-smi fails', async () => {
    mockSpawn.mockReturnValueOnce(
      createMockProcess('', 'nvidia-smi: command not found', 1)
    );

    const route = await importRoute();
    const response = await route.POST();

    expect(response.status).toBe(500);
    const body = await response.json();
    expect(body.error).toBeDefined();
  });

  it('response includes resolved CUDA version string', async () => {
    mockSpawn
      .mockReturnValueOnce(
        createMockProcess(
          "NVIDIA-SMI 535.00\nGPU Name: NVIDIA GeForce RTX 5090",
          '',
          0
        )
      )
      .mockReturnValueOnce(
        createMockProcess('', '', 0)
      );

    const route = await importRoute();
    const response = await route.POST();
    const body = await response.json();

    expect(['cu128', 'cu130']).toContain(body.cuda);
    expect(body.cuda).toBe('cu130'); // RTX 50 → cu130
  });
});

describe('parseGpuInfo', () => {
  async function getParseGpuInfo() {
    const route = await importRoute();
    return route.parseGpuInfo;
  }

  it('returns cu130 for RTX 50 series', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo("GPU Name: NVIDIA GeForce RTX 5090");
    expect(result).toEqual({
      cuda: 'cu130',
      series: 'blackwell',
      gpuName: 'NVIDIA GeForce RTX 5090',
      cudaVersion: null,
    });
  });

  it('returns cu128 for RTX 40 series', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo("GPU Name: NVIDIA GeForce RTX 4090");
    expect(result).toEqual({
      cuda: 'cu128',
      series: 'ada',
      gpuName: 'NVIDIA GeForce RTX 4090',
      cudaVersion: null,
    });
  });

  it('returns cu128 for RTX 30 series', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo("GPU Name: NVIDIA GeForce RTX 3080");
    expect(result).toEqual({
      cuda: 'cu128',
      series: 'ampere',
      gpuName: 'NVIDIA GeForce RTX 3080',
      cudaVersion: null,
    });
  });

  it('extracts CUDA toolkit version from output', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo(
      "NVIDIA-SMI 535.00\nGPU Name: NVIDIA GeForce RTX 4090\nCUDA Version: 12.8"
    );
    expect(result?.cudaVersion).toBe('12.8');
  });

  it('returns null for unsupported GPU', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo("GPU Name: NVIDIA GeForce GTX 1080");
    expect(result).toBeNull();
  });

  it('returns null for empty input', async () => {
    const parseGpuInfo = await getParseGpuInfo();
    const result = parseGpuInfo('');
    expect(result).toBeNull();
  });
});
