export interface ModelEntry {
  name: string;
  hfPath: string;        // e.g., "circlestone-labs/Anima:main:split_files/diffusion_models/anima-base-v1.0.safetensors"
  hfRepo: string;        // e.g., "circlestone-labs/Anima"
  hfFile: string;        // e.g., "split_files/diffusion_models/anima-base-v1.0.safetensors"
  localPath: string;     // e.g., "models/anima/diffusion_models/anima-base-v1.0.safetensors"
  expectedSizeBytes: number;
}

export type ModelType = 'anima' | 'flux' | 'sd3' | 'sdxl' | 'sd15' | 'hunyuan' | 'lumina';

const ANIMA_MODELS: ModelEntry[] = [
  {
    name: 'diffusion_model',
    hfRepo: 'circlestone-labs/Anima',
    hfFile: 'split_files/diffusion_models/anima-base-v1.0.safetensors',
    hfPath: 'circlestone-labs/Anima:main:split_files/diffusion_models/anima-base-v1.0.safetensors',
    localPath: 'models/anima/diffusion_models/anima-base-v1.0.safetensors',
    expectedSizeBytes: 4_180_000_000, // ~4.18 GB
  },
  {
    name: 'vae',
    hfRepo: 'circlestone-labs/Anima',
    hfFile: 'split_files/vae/qwen_image_vae.safetensors',
    hfPath: 'circlestone-labs/Anima:main:split_files/vae/qwen_image_vae.safetensors',
    localPath: 'models/anima/vae/qwen_image_vae.safetensors',
    expectedSizeBytes: 254_000_000, // ~254 MB
  },
  {
    name: 'text_encoder',
    hfRepo: 'circlestone-labs/Anima',
    hfFile: 'split_files/text_encoders/qwen_3_06b_base.safetensors',
    hfPath: 'circlestone-labs/Anima:main:split_files/text_encoders/qwen_3_06b_base.safetensors',
    localPath: 'models/anima/text_encoders/qwen_3_06b_base.safetensors',
    expectedSizeBytes: 1_200_000_000, // ~1.2 GB (estimated)
  },
];

const MODEL_MANIFESTS: Record<ModelType, ModelEntry[]> = {
  anima: ANIMA_MODELS,
  flux: [],
  sd3: [],
  sdxl: [],
  sd15: [],
  hunyuan: [],
  lumina: [],
};

export function getModelManifest(modelType: ModelType): ModelEntry[] {
  const manifest = MODEL_MANIFESTS[modelType];
  if (!manifest) {
    throw new Error(`Unknown model type: ${modelType}`);
  }
  return manifest;
}
