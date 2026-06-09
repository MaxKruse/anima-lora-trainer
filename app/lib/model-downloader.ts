import { spawn, execSync } from 'child_process';
import { ModelEntry } from './model-manifest';
import fs from 'fs';

export interface DownloadProgress {
  model: string;
  status: 'started' | 'downloading' | 'completed' | 'failed';
  progress?: number;    // 0-100
  message?: string;
  error?: string;
}

const MAX_RETRIES = 3;

/**
 * Resolve the cached file path using Python's huggingface_hub library.
 * Returns the path if the file is cached, or null if not found.
 */
export function resolveCachePath(repo: string, file: string): string | null {
  const script = `
import sys
from huggingface_hub import try_to_load_from_cache
result = try_to_load_from_cache("${repo}", "${file}", repo_type="model")
if isinstance(result, str):
    print(result, end="")
`;

  try {
    const output = execSync('python -c ' + JSON.stringify(script), {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'], // suppress stderr
    }).trim();
    if (output && fs.existsSync(output)) {
      return output;
    }
  } catch {
    // File not cached or Python not available
  }
  return null;
}

/**
 * Download a model file to the global HF cache.
 * Uses `hf download` without --local-dir so files land in ~/.cache/huggingface/hub.
 */
export async function downloadModel(
  entry: ModelEntry,
  onProgress?: (progress: DownloadProgress) => void,
  abortSignal?: AbortSignal
): Promise<void> {
  const { hfRepo, hfFile } = entry;

  console.log(`[download] === downloadModel START: ${entry.name} ===`);
  console.log(`[download]   hfRepo=${hfRepo}`);
  console.log(`[download]   hfFile=${hfFile}`);

  // Check if already cached
  const cachedPath = resolveCachePath(hfRepo, hfFile);
  if (cachedPath) {
    const stat = fs.statSync(cachedPath);
    console.log(`[download] File already cached at ${cachedPath}, size=${stat.size} bytes`);
    onProgress?.({
      model: entry.name,
      status: 'completed',
      progress: 100,
      message: `Already cached: ${entry.name}`,
    });
    console.log(`[download] === downloadModel SUCCESS (cached): ${entry.name} ===`);
    return;
  }

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    console.log(`[download] Attempt ${attempt}/${MAX_RETRIES} for ${entry.name}`);

    if (abortSignal?.aborted) {
      console.log(`[download] Aborted before attempt ${attempt}: ${entry.name}`);
      throw new Error(`Download aborted: ${entry.name}`);
    }

    onProgress?.({
      model: entry.name,
      status: attempt > 1 ? 'downloading' : 'started',
      message: `Downloading ${entry.name} (attempt ${attempt}/${MAX_RETRIES})...`,
    });

    try {
      console.log(`[download] Spawning: hf download ${hfRepo} ${hfFile} --quiet`);

      await runHuggingfaceDownload(hfRepo, hfFile, abortSignal, (progress) => {
        onProgress?.({ ...progress, model: entry.name });
      });

      console.log(`[download] hf process exited successfully for ${entry.name}`);

      // Resolve the cached path after download
      const resolvedPath = resolveCachePath(hfRepo, hfFile);
      if (resolvedPath) {
        const stat = fs.statSync(resolvedPath);
        console.log(`[download] Resolved cache path: ${resolvedPath}, size=${stat.size} bytes`);
      } else {
        console.error(`[download] WARNING: hf exited 0 but could not resolve cache path!`);
      }

      onProgress?.({
        model: entry.name,
        status: 'completed',
        progress: 100,
        message: `Downloaded ${entry.name} successfully`,
      });

      console.log(`[download] === downloadModel SUCCESS: ${entry.name} ===`);
      return;
    } catch (error: any) {
      const errorMsg = error.message || 'Unknown error';
      console.error(`[download] Attempt ${attempt} FAILED for ${entry.name}: ${errorMsg}`);

      if (attempt === MAX_RETRIES) {
        console.error(`[download] All ${MAX_RETRIES} attempts exhausted for ${entry.name}`);
        onProgress?.({
          model: entry.name,
          status: 'failed',
          error: `Failed after ${MAX_RETRIES} attempts: ${errorMsg}`,
        });
        throw new Error(`Download failed for ${entry.name}: ${errorMsg}`);
      }

      onProgress?.({
        model: entry.name,
        status: 'downloading',
        message: `Attempt ${attempt} failed: ${errorMsg}. Retrying...`,
      });
    }
  }
}

function runHuggingfaceDownload(
  repo: string,
  file: string,
  signal?: AbortSignal,
  onProgress?: (progress: DownloadProgress) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log(`[hf:spawn] Spawning: hf download ${repo} ${file} --quiet`);

    // Force UTF-8 encoding for Python's stdout/stderr — fixes Windows cp1252
    // "charmap codec can't encode character '\u2713'" crash from hf CLI progress output
    // Disable HF hub's own progress bars so we parse stderr ourselves
    const spawnEnv = {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
      HF_HUB_DISABLE_PROGRESS_BARS: 'false', // We want progress in stderr for parsing
    };

    // Use --quiet so stdout is just the cached file path (no extra messages)
    // Progress/tqdm output goes to stderr which we parse
    const proc = spawn(
      'hf',
      ['download', repo, file, '--quiet'],
      { shell: false, env: spawnEnv }
    );

    console.log(`[hf:spawn] Process spawned, pid=${proc.pid}`);

    let stderr = '';
    let stdout = '';
    let lastReportedProgress = -1;

    proc.stdout.on('data', (data) => {
      const chunk = data.toString();
      stdout += chunk;
    });

    proc.stderr.on('data', (data) => {
      const chunk = data.toString();
      stderr += chunk;

      // Strip ANSI escape codes and \r for progress parsing
      const cleaned = chunk.replace(/\x1b\[[0-9;]*m/g, '').replace(/\r/g, '');

      // Parse tqdm-style progress: "Downloading files: 42%|██████ | 1.20G/3.50G"
      const progressMatch = cleaned.match(/(\d+)%/);
      if (progressMatch) {
        const progress = parseInt(progressMatch[1], 10);
        if (progress !== lastReportedProgress && progress <= 100) {
          lastReportedProgress = progress;
          console.log(`[hf:progress] ${progress}%`);
          onProgress?.({
            model: '', // caller sets model name
            status: 'downloading',
            progress,
          });
        }
      }
    });

    proc.on('close', (code) => {
      console.log(`[hf:close] Process exited with code=${code}`);
      if (stdout.trim()) {
        console.log(`[hf:close] stdout (cache path): ${stdout.trim()}`);
      }
      if (stderr) {
        console.log(`[hf:close] Full stderr (${stderr.length} chars): ${stderr.replace(/\n/g, ' | ').slice(0, 200)}`);
      }

      if (code === 0) {
        console.log(`[hf:close] Success — resolving`);
        resolve();
      } else {
        const errMsg = stderr || `hf exited with code ${code}`;
        console.error(`[hf:close] FAILURE — rejecting with: ${errMsg}`);
        reject(new Error(errMsg));
      }
    });

    proc.on('error', (err) => {
      console.error(`[hf:error] Spawn error: ${err.message || err}`);
      reject(err);
    });

    if (signal) {
      signal.addEventListener('abort', () => {
        console.log(`[hf:abort] Aborting hf process pid=${proc.pid}`);
        proc.kill();
        reject(new Error('Download aborted'));
      });
    }
  });
}

/**
 * Check if a model file exists in the HF cache.
 * Returns cache info if found, or { exists: false } if not.
 */
export async function checkModelStatus(entry: ModelEntry): Promise<{
  exists: boolean;
  sizeBytes?: number;
  downloadPercent?: number;
  cachePath?: string;
}> {
  const cachedPath = resolveCachePath(entry.hfRepo, entry.hfFile);

  console.log(`[status] checkModelStatus: ${entry.name}`);
  console.log(`[status]   hfRepo=${entry.hfRepo}`);
  console.log(`[status]   hfFile=${entry.hfFile}`);

  if (cachedPath) {
    const stat = fs.statSync(cachedPath);
    const percent = entry.expectedSizeBytes > 0
      ? Math.round((stat.size / entry.expectedSizeBytes) * 100)
      : 0;
    console.log(`[status]   Cached at: ${cachedPath}`);
    console.log(`[status]   size=${stat.size} bytes, expected=${entry.expectedSizeBytes} bytes, percent=${Math.min(percent, 100)}%`);
    return {
      exists: true,
      sizeBytes: stat.size,
      downloadPercent: Math.min(percent, 100),
      cachePath: cachedPath,
    };
  }

  console.log(`[status]   NOT in cache`);
  return { exists: false };
}
