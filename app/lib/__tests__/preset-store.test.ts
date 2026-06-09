import { describe, it, expect, vi, beforeEach } from 'vitest';
import fs from 'fs';

// --- Mock fs ---
const mockExistsSync = vi.fn(() => false);
const mockReadFileSync = vi.fn();
const mockWriteFileSync = vi.fn();
const mockMkdirSync = vi.fn();
const fsMocks = {
  existsSync: (...args: any[]) => mockExistsSync(...args),
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  mkdirSync: (...args: any[]) => mockMkdirSync(...args),
};
vi.mock('fs', () => ({
  default: fsMocks,
  ...fsMocks,
}));

// --- Mock path ---
const pathMocks = {
  join: (...args: string[]) => args.join('/'),
  resolve: (...args: string[]) => args.join('/'),
};
vi.mock('path', () => ({
  default: pathMocks,
  ...pathMocks,
}));

async function importPresetStore() {
  const mod = await import('../preset-store');
  return mod.PresetStore;
}

const sampleParams = {
  networkDim: 32,
  networkAlpha: 16,
  learningRate: 1e-4,
  batchSize: 1,
  epochs: 10,
};

describe('PresetStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockExistsSync.mockReturnValue(false);
  });

  it('savePreset(name, params) stores preset', async () => {
    const PresetStore = await importPresetStore();
    const store = new PresetStore();

    store.savePreset('my-config', sampleParams);

    expect(mockWriteFileSync).toHaveBeenCalled();
    const writtenData = JSON.parse(mockWriteFileSync.mock.calls[0][1] as string);
    expect(writtenData['my-config']).toEqual(sampleParams);
  });

  it('loadPreset(name) returns stored params', async () => {
    const storedData = { 'my-config': sampleParams };
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue(JSON.stringify(storedData));

    const PresetStore = await importPresetStore();
    const store = new PresetStore();

    const loaded = store.loadPreset('my-config');

    expect(loaded).toEqual(sampleParams);
  });

  it('listPresets() returns all preset names', async () => {
    const storedData = {
      'my-config': sampleParams,
      'another-config': { ...sampleParams, epochs: 20 },
    };
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue(JSON.stringify(storedData));

    const PresetStore = await importPresetStore();
    const store = new PresetStore();

    const names = store.listPresets();

    expect(names).toContain('my-config');
    expect(names).toContain('another-config');
    expect(names).toHaveLength(2);
  });

  it('overwriting preset name replaces old data', async () => {
    const oldData = { 'my-config': { networkDim: 16 } };
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue(JSON.stringify(oldData));

    const PresetStore = await importPresetStore();
    const store = new PresetStore();

    store.savePreset('my-config', sampleParams);

    // Should have written updated data
    expect(mockWriteFileSync).toHaveBeenCalled();
    const writtenData = JSON.parse(mockWriteFileSync.mock.calls[0][1] as string);
    expect(writtenData['my-config']).toEqual(sampleParams);
  });

  it('deleting preset removes it', async () => {
    const storedData = {
      'my-config': sampleParams,
      'another-config': { ...sampleParams, epochs: 20 },
    };
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue(JSON.stringify(storedData));

    const PresetStore = await importPresetStore();
    const store = new PresetStore();

    store.deletePreset('my-config');

    expect(mockWriteFileSync).toHaveBeenCalled();
    const writtenData = JSON.parse(mockWriteFileSync.mock.calls[0][1] as string);
    expect(writtenData['my-config']).toBeUndefined();
    expect(writtenData['another-config']).toBeDefined();
  });
});
