import { spawn, execSync } from 'child_process';
import { ModelEntry } from './model-manifest';
import fs from 'fs';
import os from 'os';
import path from 'path';

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
 * Python script that downloads a file via hf_hub_download and reports progress on stdout as JSON lines.
 * Each line: {"type": "progress", "percent": 42} or {"type": "done", "path": "..."} or {"type": "error", "message": "..."}
 */
const DOWNLOAD_SCRIPT = `
import sys, json, os
from huggingface_hub import hf_hub_download

class ProgressTqdm:
    """Custom tqdm that reports progress as JSON lines on stdout."""
    def __init__(self, total=None, unit=None, unit_scale=None, desc=None, **kwargs):
        self.total = total
        self.n = 0
        self.last_reported = -1

    def update(self, n=1):
        self.n += n
        if self.total and self.total > 0:
            pct = min(int((self.n / self.total) * 100), 100)
            if pct != self.last_reported:
                self.last_reported = pct
                json.dump({"type": "progress", "percent": pct}, sys.stdout)
                sys.stdout.write("\\n")
                sys.stdout.flush()

    def close(self):
        # Ensure 100% is reported if we reached the end
        if self.total and self.total > 0 and self.last_reported < 100:
            json.dump({"type": "progress", "percent": 100}, sys.stdout)
            sys.stdout.write("\\n")
            sys.stdout.flush()

    def set_description(self, desc=None):
        pass

repo_id = sys.argv[1]
filename = sys.argv[2]

try:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        tqdm_class=ProgressTqdm,
    )
    json.dump({"type": "done", "path": path}, sys.stdout)
    sys.stdout.write("\\n")
    sys.stdout.flush()
except Exception as e:
    json.dump({"type": "error", "message": str(e)}, sys.stdout)
    sys.stdout.write("\\n")
    sys.stdout.flush()
    sys.exit(1)
`;

/**
 * Download a model file to the global HF cache.
 * Uses Python's hf_hub_download directly for reliable progress reporting.
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
      console.log(`[download] Spawning Python hf_hub_download for ${entry.name}`);

      await runPythonDownload(hfRepo, hfFile, abortSignal, (progress) => {
        onProgress?.({ ...progress, model: entry.name });
      });

      console.log(`[download] Python download completed for ${entry.name}`);

      // Resolve the cached path after download
      const resolvedPath = resolveCachePath(hfRepo, hfFile);
      if (resolvedPath) {
        const stat = fs.statSync(resolvedPath);
        console.log(`[download] Resolved cache path: ${resolvedPath}, size=${stat.size} bytes`);
      } else {
        console.error(`[download] WARNING: download succeeded but could not resolve cache path!`);
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

function runPythonDownload(
  repo: string,
  file: string,
  signal?: AbortSignal,
  onProgress?: (progress: DownloadProgress) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log(`[py:spawn] Spawning: python <script> ${repo} ${file}`);

    // Force UTF-8 and unbuffered output so progress JSON lines arrive in real-time
    const spawnEnv = {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
      PYTHONUNBUFFERED: '1',
    };

    // Write the script to a temp file to avoid shell escaping issues
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hf-dl-'));
    const scriptFile = path.join(tmpDir, 'download.py');
    fs.writeFileSync(scriptFile, DOWNLOAD_SCRIPT, 'utf-8');

    const proc = spawn('python', [scriptFile, repo, file], {
      shell: false,
      env: spawnEnv,
    });

    console.log(`[py:spawn] Process spawned, pid=${proc.pid}`);

    let stderr = '';
    let lastReportedProgress = -1;
    let buffer = '';

    // Progress JSON lines come on stdout
    proc.stdout.on('data', (data) => {
      const chunk = data.toString();
      buffer += chunk;

      // Split on newline characters
      const parts = buffer.split(/\r?\n/);
      // Keep the last (possibly incomplete) part in the buffer
      buffer = parts.pop() || '';

      for (const line of parts) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        try {
          const msg = JSON.parse(trimmed);

          if (msg.type === 'progress') {
            const pct = msg.percent;
            if (pct !== lastReportedProgress && pct <= 100) {
              lastReportedProgress = pct;
              console.log(`[py:progress] ${pct}%`);
              onProgress?.({
                model: '',
                status: 'downloading',
                progress: pct,
              });
            }
          } else if (msg.type === 'done') {
            console.log(`[py:done] File cached at: ${msg.path}`);
          } else if (msg.type === 'error') {
            console.error(`[py:error] ${msg.message}`);
            reject(new Error(msg.message));
            return;
          }
        } catch {
          // Not a JSON line, ignore
        }
      }
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      console.log(`[py:close] Process exited with code=${code}`);
      if (stderr) {
        console.log(`[py:close] stderr (${stderr.length} chars): ${stderr.slice(0, 200)}`);
      }

      if (code === 0) {
        console.log(`[py:close] Success — resolving`);
        // Clean up script file
        try { fs.unlinkSync(scriptFile); fs.rmdirSync(tmpDir); } catch { /* ignore */ }
        resolve();
      } else {
        const errMsg = stderr || `Python download exited with code ${code}`;
        console.error(`[py:close] FAILURE — rejecting with: ${errMsg}`);
        try { fs.unlinkSync(scriptFile); fs.rmdirSync(tmpDir); } catch { /* ignore */ }
        reject(new Error(errMsg));
      }
    });

    proc.on('error', (err) => {
      console.error(`[py:error] Spawn error: ${err.message || err}`);
      try { fs.unlinkSync(scriptFile); fs.rmdirSync(tmpDir); } catch { /* ignore */ }
      reject(err);
    });

    if (signal) {
      signal.addEventListener('abort', () => {
        console.log(`[py:abort] Aborting Python process pid=${proc.pid}`);
        proc.kill();
        try { fs.unlinkSync(scriptFile); fs.rmdirSync(tmpDir); } catch { /* ignore */ }
        reject(new Error('Download aborted'));
      });
    }
  });
}

/**
 * Check if a model file exists in the HF cache.
 * Returns cache info if found, or { exists: false } if not.
 */
export async function checkModelStatus(entry: ModelEntry & { expectedSizeBytes?: number }): Promise<{
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
    const expectedSize = entry.expectedSizeBytes ?? 0;
    const percent = expectedSize > 0
      ? Math.round((stat.size / expectedSize) * 100)
      : 0;
    console.log(`[status]   Cached at: ${cachedPath}`);
    console.log(`[status]   size=${stat.size} bytes, expected=${expectedSize} bytes, percent=${Math.min(percent, 100)}%`);
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
