export interface ModelEntry {
  name: string;
  hfRepo: string;
  hfFile: string;
  localPath: string;
}

export interface ResolvedModelEntry extends ModelEntry {
  expectedSizeBytes: number;
}

export type ModelType = 'anima' | 'flux' | 'sd3' | 'sdxl' | 'sd15' | 'hunyuan' | 'lumina';

const ANIMA_MODELS: ModelEntry[] = [
  {
    name: 'diffusion_model',
    hfRepo: 'circlestone-labs/Anima',
    hfFile: 'split_files/diffusion_models/anima-base-v1.0.safetensors',
    localPath: 'models/anima/diffusion_models/anima-base-v1.0.safetensors',
  },
  {
    name: 'vae',
    hfRepo: 'circlestone-labs/Anima',
    hfFile: 'split_files/vae/qwen_image_vae.safetensors',
    localPath: 'models/anima/vae/qwen_image_vae.safetensors',
  },
  {
    name: 'text_encoder',
    hfRepo: 'circlestone-labs/Anima',
    hfFile: 'split_files/text_encoders/qwen_3_06b_base.safetensors',
    localPath: 'models/anima/text_encoders/qwen_3_06b_base.safetensors',
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

/** Cache: repo → (file → size) map, with expiry timestamp */
interface CacheEntry {
  sizes: Map<string, number>;
  expiresAt: number;
}
const sizeCache = new Map<string, CacheEntry>();
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

/**
 * Fetch file sizes for a repo from the HuggingFace REST API.
 * Returns a map of file path → size in bytes.
 */
async function fetchRepoFileSizes(repo: string): Promise<Map<string, number>> {
  const cached = sizeCache.get(repo);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.sizes;
  }

  // Don't encode the full repo path — HF API expects owner/repo with literal /
  const url = `https://huggingface.co/api/models/${repo}/tree/main?recursive=true`;
  const response = await fetch(url, {
    headers: { 'User-Agent': 'lora-matrix-trainer/1.0' },
    cache: 'no-store', // Bypass Next.js fetch cache
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch file sizes for ${repo}: ${response.status} ${response.statusText}`);
  }

  const entries = (await response.json()) as Array<{ path: string; size?: number; lfs?: { size: number }; type: string }>;
  const sizes = new Map<string, number>();

  for (const entry of entries) {
    if (entry.type === 'file' && entry.path) {
      // lfs.size is the actual file size for LFS-tracked files
      sizes.set(entry.path, entry.lfs?.size ?? entry.size ?? 0);
    }
  }

  sizeCache.set(repo, { sizes, expiresAt: Date.now() + CACHE_TTL_MS });
  return sizes;
}

/** Clear the size cache. Used for testing. */
export function _clearSizeCache(): void {
  sizeCache.clear();
}

/**
 * Get the base manifest (no sizes). Use `getResolvedModelManifest` for entries with sizes.
 */
export function getModelManifest(modelType: ModelType): ModelEntry[] {
  const manifest = MODEL_MANIFESTS[modelType];
  if (!manifest) {
    throw new Error(`Unknown model type: ${modelType}`);
  }
  return manifest;
}

/**
 * Get the manifest with actual file sizes resolved from the HF API.
 * Sizes are cached for CACHE_TTL_MS.
 */
export async function getResolvedModelManifest(modelType: ModelType): Promise<ResolvedModelEntry[]> {
  const entries = getModelManifest(modelType);
  if (entries.length === 0) return [];

  // Group by repo to minimize API calls
  const repos = [...new Set(entries.map(e => e.hfRepo))];
  const repoSizes = new Map<string, Map<string, number>>();

  for (const repo of repos) {
    try {
      repoSizes.set(repo, await fetchRepoFileSizes(repo));
    } catch (error: any) {
      console.warn(`Could not fetch file sizes for ${repo}: ${error.message} — sizes will be 0 until API recovers`);
    }
  }

  return entries.map((entry) => {
    const sizes = repoSizes.get(entry.hfRepo);
    const size = sizes?.get(entry.hfFile) ?? 0;

    if (size === 0) {
      console.warn(`Could not resolve size for ${entry.hfRepo}/${entry.hfFile} — using 0`);
    }

    return {
      ...entry,
      expectedSizeBytes: size,
    };
  });
}
