import { z } from 'zod';

export const trainingSchema = z.object({
  // Network parameters
  networkDim: z.number().int().min(1),
  networkAlpha: z.number().min(0),

  // Training parameters
  learningRate: z.number().min(0),
  batchSize: z.number().int().min(1),
  epochs: z.number().int().min(1),

  // Optimizer
  optimizer: z.enum(['AdamW8Bit', 'AdamW', 'Prodigy', 'Lion', 'Adafactor']),

  // LR Scheduler
  scheduler: z.enum(['constant', 'cosine', 'linear', 'constant_with_warmup', 'cosine_with_restarts']),

  // Data
  trainingImages: z.string().min(1),
  loraName: z.string().min(1),

  // Precision
  mixedPrecision: z.enum(['fp16', 'bf16', 'no']),

  // Sampling
  timestepSampling: z.enum(['sigma', 'uniform', 'sigmoid', 'shift', 'flux_shift']),

  // Optimizations
  gradientCheckpointing: z.boolean().default(true),
  cacheLatents: z.boolean().default(true),
  cacheTextEncoder: z.boolean().default(true),
});

export type TrainingParams = z.infer<typeof trainingSchema>;
