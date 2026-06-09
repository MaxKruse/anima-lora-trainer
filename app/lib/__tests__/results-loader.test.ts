import { describe, it, expect, vi, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';

// --- Mock fs ---
const mockReadFileSync = vi.fn();
const mockExistsSync = vi.fn(() => true);
vi.mock('fs', () => ({
  default: {
    readFileSync: (...args: any[]) => mockReadFileSync(...args),
    existsSync: (...args: any[]) => mockExistsSync(...args),
  },
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  existsSync: (...args: any[]) => mockExistsSync(...args),
}));

// --- Mock path ---
vi.mock('path', () => ({
  default: {
    join: (...args: string[]) => args.join('/'),
    resolve: (...args: string[]) => args.join('/'),
  },
  join: (...args: string[]) => args.join('/'),
  resolve: (...args: string[]) => args.join('/'),
}));

async function importResultsLoader() {
  const mod = await import('../results-loader');
  return mod.loadResults;
}

const sampleManifest = {
  permutations: [
    {
      index: 0,
      params: { 'network-dim': 32, 'network-alpha': 16, 'learning-rate': 0.0001 },
      status: 'completed',
      output_files: ['lora-000001.safetensors'],
      error: null,
    },
    {
      index: 1,
      params: { 'network-dim': 64, 'network-alpha': 32, 'learning-rate': 0.0001 },
      status: 'completed',
      output_files: ['lora-000002.safetensors'],
      error: null,
    },
    {
      index: 2,
      params: { 'network-dim': 32, 'network-alpha': 16, 'learning-rate': 0.001 },
      status: 'failed',
      output_files: [],
      error: 'OOM error',
    },
  ],
  total: 3,
  completed: 2,
  failed: 1,
};

const sampleEvaluation = {
  prompt: 'cat dog bird',
  seed: 42,
  total: 3,
  completed: 2,
  failed: 1,
  results: [
    {
      perm_name: 'anima-network-alpha_16-network-dim_32-learning-rate_1e-4',
      lora_file: 'lora-000001.safetensors',
      image_file: 'eval_perm-a.png',
      status: 'completed',
      inference_time_ms: 1500,
    },
    {
      perm_name: 'anima-network-alpha_32-network-dim_64-learning-rate_1e-4',
      lora_file: 'lora-000002.safetensors',
      image_file: 'eval_perm-b.png',
      status: 'completed',
      inference_time_ms: 1600,
    },
    {
      perm_name: 'anima-network-alpha_16-network-dim_32-learning-rate_1e-3',
      lora_file: 'lora-000003.safetensors',
      image_file: null,
      status: 'failed',
      error: 'sd-cli crash',
      inference_time_ms: 0,
    },
  ],
};

describe('loadResults', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('loads manifest and evaluation.json from run directory', async () => {
    mockReadFileSync
      .mockReturnValueOnce(JSON.stringify(sampleManifest))
      .mockReturnValueOnce(JSON.stringify(sampleEvaluation));

    const loadResults = await importResultsLoader();
    const results = loadResults('/path/to/run');

    expect(results).toBeDefined();
    expect(mockReadFileSync).toHaveBeenCalledTimes(2);
  });

  it('merges permutation params with evaluation results', async () => {
    mockReadFileSync
      .mockReturnValueOnce(JSON.stringify(sampleManifest))
      .mockReturnValueOnce(JSON.stringify(sampleEvaluation));

    const loadResults = await importResultsLoader();
    const results = loadResults('/path/to/run');

    expect(results).toHaveLength(3);

    // First result should have both params and eval data
    const first = results[0];
    expect(first.params).toHaveProperty('network-dim', 32);
    expect(first.params).toHaveProperty('network-alpha', 16);
    expect(first.loraFile).toBe('lora-000001.safetensors');
    expect(first.imageFile).toBe('eval_perm-a.png');
    expect(first.status).toBe('completed');
    expect(first.inferenceTimeMs).toBe(1500);
  });

  it('returns array of { params, loraFile, imageFile, status, inferenceTimeMs }', async () => {
    mockReadFileSync
      .mockReturnValueOnce(JSON.stringify(sampleManifest))
      .mockReturnValueOnce(JSON.stringify(sampleEvaluation));

    const loadResults = await importResultsLoader();
    const results = loadResults('/path/to/run');

    for (const entry of results) {
      expect(entry).toHaveProperty('params');
      expect(entry).toHaveProperty('loraFile');
      expect(entry).toHaveProperty('imageFile');
      expect(entry).toHaveProperty('status');
      expect(entry).toHaveProperty('inferenceTimeMs');
    }
  });

  it('handles missing evaluation.json (returns results without images)', async () => {
    // existsSync is only called for evaluation.json — return false
    mockExistsSync.mockReturnValue(false);
    mockReadFileSync.mockReturnValueOnce(JSON.stringify(sampleManifest));

    const loadResults = await importResultsLoader();
    const results = loadResults('/path/to/run');

    expect(results).toHaveLength(3);

    // Without evaluation data, imageFile should be null/undefined
    for (const entry of results) {
      expect(entry.imageFile).toBeNull();
      expect(entry.inferenceTimeMs).toBeNull();
    }
  });
});
