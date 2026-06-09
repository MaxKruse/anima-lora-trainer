import { describe, it, expect } from 'vitest';

async function importSchema() {
  const mod = await import('../training-schema');
  return mod.trainingSchema;
}

describe('trainingSchema', () => {
  const validParams = {
    networkDim: 8,
    networkAlpha: 1,
    learningRate: 1e-4,
    batchSize: 1,
    epochs: 10,
    optimizer: 'AdamW8Bit',
    scheduler: 'cosine',
    trainingImages: '/path/to/images',
    loraName: 'my-lora',
    mixedPrecision: 'bf16',
    timestepSampling: 'sigmoid',
    gradientCheckpointing: true,
    cacheLatents: true,
    cacheTextEncoder: true,
  };

  it('accepts valid single-run parameter set', async () => {
    const schema = await importSchema();
    const result = schema.safeParse(validParams);
    expect(result.success).toBe(true);
  });

  it('rejects missing required field: network_dim', async () => {
    const schema = await importSchema();
    const params = { ...validParams };
    delete (params as any).networkDim;
    const result = schema.safeParse(params);
    expect(result.success).toBe(false);
  });

  it('rejects missing required field: learning_rate', async () => {
    const schema = await importSchema();
    const params = { ...validParams };
    delete (params as any).learningRate;
    const result = schema.safeParse(params);
    expect(result.success).toBe(false);
  });

  it('rejects missing required field: epochs', async () => {
    const schema = await importSchema();
    const params = { ...validParams };
    delete (params as any).epochs;
    const result = schema.safeParse(params);
    expect(result.success).toBe(false);
  });

  it('rejects negative network_dim', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, networkDim: -1 });
    expect(result.success).toBe(false);
  });

  it('rejects zero epochs', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, epochs: 0 });
    expect(result.success).toBe(false);
  });

  it('rejects negative learning_rate', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, learningRate: -1e-4 });
    expect(result.success).toBe(false);
  });

  it('rejects negative batch_size', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, batchSize: -1 });
    expect(result.success).toBe(false);
  });

  it('accepts all optimizer types from spec', async () => {
    const schema = await importSchema();
    const optimizers = ['AdamW8Bit', 'AdamW', 'Prodigy', 'Lion', 'Adafactor'];
    for (const opt of optimizers) {
      const result = schema.safeParse({ ...validParams, optimizer: opt });
      expect(result.success, `Optimizer ${opt} should be valid`).toBe(true);
    }
  });

  it('rejects invalid optimizer', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, optimizer: 'InvalidOpt' });
    expect(result.success).toBe(false);
  });

  it('accepts all scheduler types from spec', async () => {
    const schema = await importSchema();
    const schedulers = ['constant', 'cosine', 'linear', 'constant_with_warmup', 'cosine_with_restarts'];
    for (const sched of schedulers) {
      const result = schema.safeParse({ ...validParams, scheduler: sched });
      expect(result.success, `Scheduler ${sched} should be valid`).toBe(true);
    }
  });

  it('rejects invalid scheduler', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, scheduler: 'invalid' });
    expect(result.success).toBe(false);
  });

  it('accepts all mixed precision options', async () => {
    const schema = await importSchema();
    const precisions = ['fp16', 'bf16', 'no'];
    for (const prec of precisions) {
      const result = schema.safeParse({ ...validParams, mixedPrecision: prec });
      expect(result.success, `Precision ${prec} should be valid`).toBe(true);
    }
  });

  it('accepts all timestep sampling options', async () => {
    const schema = await importSchema();
    const samplings = ['sigma', 'uniform', 'sigmoid', 'shift', 'flux_shift'];
    for (const ts of samplings) {
      const result = schema.safeParse({ ...validParams, timestepSampling: ts });
      expect(result.success, `Timestep sampling ${ts} should be valid`).toBe(true);
    }
  });

  // --- resolution ---
  it('defaults resolution to 1024 when omitted', async () => {
    const schema = await importSchema();
    const result = schema.safeParse(validParams);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.resolution).toBe(1024);
    }
  });

  it('accepts valid resolution values', async () => {
    const schema = await importSchema();
    for (const res of [256, 512, 768, 1024, 2048]) {
      const result = schema.safeParse({ ...validParams, resolution: res });
      expect(result.success, `Resolution ${res} should be valid`).toBe(true);
      if (result.success) expect(result.data.resolution).toBe(res);
    }
  });

  it('rejects resolution below 256', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, resolution: 128 });
    expect(result.success).toBe(false);
  });

  it('rejects non-integer resolution', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, resolution: 768.5 });
    expect(result.success).toBe(false);
  });

  // --- maxSteps (optional) ---
  it('accepts valid maxSteps', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, maxSteps: 500 });
    expect(result.success).toBe(true);
    if (result.success) expect(result.data.maxSteps).toBe(500);
  });

  it('accepts omitted maxSteps', async () => {
    const schema = await importSchema();
    const params = { ...validParams };
    delete (params as any).maxSteps;
    const result = schema.safeParse(params);
    expect(result.success).toBe(true);
  });

  it('rejects maxSteps below 1', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, maxSteps: 0 });
    expect(result.success).toBe(false);
  });

  // --- repeats (optional) ---
  it('accepts valid repeats', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, repeats: 10 });
    expect(result.success).toBe(true);
    if (result.success) expect(result.data.repeats).toBe(10);
  });

  it('accepts omitted repeats', async () => {
    const schema = await importSchema();
    const params = { ...validParams };
    delete (params as any).repeats;
    const result = schema.safeParse(params);
    expect(result.success).toBe(true);
  });

  it('rejects repeats below 1', async () => {
    const schema = await importSchema();
    const result = schema.safeParse({ ...validParams, repeats: 0 });
    expect(result.success).toBe(false);
  });
});
