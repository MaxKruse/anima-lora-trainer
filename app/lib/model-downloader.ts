import { spawn } from 'child_process';
import { ModelEntry, ResolvedModelEntry } from './model-manifest';
import fs from 'fs';
import os from 'os';
import path from 'path';

export interface DownloadProgress {
  model: string;
  status: 'started' | 'downloading' | 'completed' | 'failed';
  progress?: number;    // 0-100
  downloaded?: number;  // bytes downloaded so far
  message?: string;
  error?: string;
}

const MAX_RETRIES = 3;
const MODELS_DIR = path.join(process.cwd(), 'models');

/**
 * Python script that downloads a file from HuggingFace via direct HTTP.
 * Reports progress on stdout as JSON lines, debug on stderr.
 */
const DOWNLOAD_SCRIPT = `
import sys, json, os, urllib.request, urllib.error

def log(msg):
    print(f"[dl] {msg}", file=sys.stderr, flush=True)

def report(obj):
    json.dump(obj, sys.stdout)
    sys.stdout.write("\\n")
    sys.stdout.flush()

url = sys.argv[1]
dest = sys.argv[2]

log(f"Downloading to: {dest}")

# Ensure destination directory exists
os.makedirs(os.path.dirname(dest), exist_ok=True)

# Remove partial file from previous failed attempt
if os.path.exists(dest):
    log(f"Removing existing partial file: {dest}")
    os.remove(dest)

try:
    req = urllib.request.Request(url)
    # Add a User-Agent to avoid 403 from HF
    req.add_header('User-Agent', 'lora-matrix-trainer/1.0')
    
    response = urllib.request.urlopen(req, timeout=30)
    total = int(response.headers.get('Content-Length', 0))
    log(f"Content-Length: {total}")
    
    report({"type": "start", "total": total})
    
    downloaded = 0
    chunk_size = 8 * 1024 * 1024  # 8MB chunks
    
    with open(dest, 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            downloaded += len(chunk)
            f.write(chunk)
            
            pct = 0
            if total > 0:
                pct = min(int((downloaded / total) * 100), 100)
            
            report({"type": "progress", "percent": pct, "downloaded": downloaded})
    
    log(f"Download complete: {downloaded} bytes")
    report({"type": "done", "path": dest, "size": downloaded})
except urllib.error.HTTPError as e:
    msg = f"HTTP {e.code}: {e.reason}"
    log(f"ERROR: {msg}")
    report({"type": "error", "message": msg})
    sys.exit(1)
except urllib.error.URLError as e:
    msg = f"URL error: {e.reason}"
    log(f"ERROR: {msg}")
    report({"type": "error", "message": msg})
    sys.exit(1)
except Exception as e:
    msg = f"{type(e).__name__}: {e}"
    log(f"ERROR: {msg}")
    report({"type": "error", "message": msg})
    sys.exit(1)
`;

/**
 * Build the direct download URL for a HuggingFace file.
 */
function buildHfDownloadUrl(repo: string, file: string): string {
  return `https://huggingface.co/${repo}/resolve/main/${file}`;
}

/**
 * Get the local path where a model file should be stored.
 */
function getLocalPath(entry: ModelEntry): string {
  // Extract just the filename from the HF path
  const fileName = path.basename(entry.hfFile);
  return path.join(MODELS_DIR, entry.name, fileName);
}

/**
 * Check if a model file exists locally.
 */
export function checkLocalModel(entry: ModelEntry & { expectedSizeBytes?: number }): {
  exists: boolean;
  sizeBytes?: number;
  downloadPercent?: number;
  localPath?: string;
} {
  const localPath = getLocalPath(entry);
  
  if (fs.existsSync(localPath)) {
    const stat = fs.statSync(localPath);
    const expectedSize = entry.expectedSizeBytes ?? 0;
    const percent = expectedSize > 0
      ? Math.round((stat.size / expectedSize) * 100)
      : 0;
    
    console.log(`[status]   Local: ${localPath}`);
    console.log(`[status]   size=${stat.size} bytes, expected=${expectedSize} bytes, percent=${Math.min(percent, 100)}%`);
    
    return {
      exists: true,
      sizeBytes: stat.size,
      downloadPercent: Math.min(percent, 100),
      localPath,
    };
  }
  
  console.log(`[status]   NOT found locally`);
  return { exists: false };
}

/**
 * Download a model file to the local models directory.
 */
export async function downloadModel(
  entry: ModelEntry,
  onProgress?: (progress: DownloadProgress) => void,
  abortSignal?: AbortSignal
): Promise<void> {
  const { hfRepo, hfFile } = entry;
  const url = buildHfDownloadUrl(hfRepo, hfFile);
  const localPath = getLocalPath(entry);

  console.log(`[download] === downloadModel START: ${entry.name} ===`);
  console.log(`[download]   url=${url}`);
  console.log(`[download]   dest=${localPath}`);

  // Ensure models directory exists
  fs.mkdirSync(MODELS_DIR, { recursive: true });

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
      console.log(`[download] Spawning Python HTTP download for ${entry.name}`);

      const downloadedPath = await runPythonDownload(url, localPath, abortSignal, (progress) => {
        onProgress?.({ ...progress, model: entry.name });
      });

      console.log(`[download] Python download completed for ${entry.name}`);

      if (downloadedPath && fs.existsSync(downloadedPath)) {
        const stat = fs.statSync(downloadedPath);
        console.log(`[download] Downloaded file: ${downloadedPath}, size=${stat.size} bytes`);
      } else {
        console.error(`[download] WARNING: download succeeded but file not found at ${downloadedPath}!`);
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
  url: string,
  dest: string,
  signal?: AbortSignal,
  onProgress?: (progress: DownloadProgress) => void
): Promise<string | null> {
  return new Promise((resolve, reject) => {
    console.log(`[py:spawn] Spawning: python <script> ${url} ${dest}`);

    const spawnEnv = {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
    };

    // Write the script to a temp file
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hf-dl-'));
    const scriptFile = path.join(tmpDir, 'download.py');
    fs.writeFileSync(scriptFile, DOWNLOAD_SCRIPT, 'utf-8');

    const proc = spawn('python', [scriptFile, url, dest], {
      shell: false,
      env: spawnEnv,
    });

    console.log(`[py:spawn] Process spawned, pid=${proc.pid}`);

    let stderr = '';
    let lastReportedPct = -1;
    let lastReportedBytes = -1;
    let buffer = '';
    let downloadedPath: string | null = null;

    proc.stdout.on('data', (data) => {
      const chunk = data.toString();
      buffer += chunk;

      const parts = buffer.split(/\r?\n/);
      buffer = parts.pop() || '';

      for (const line of parts) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        try {
          const msg = JSON.parse(trimmed);

          if (msg.type === 'progress') {
            const pct = msg.percent;
            const downloaded = msg.downloaded;
            const pctChanged = pct !== lastReportedPct;
            const bytesChanged = downloaded !== undefined && downloaded !== lastReportedBytes;
            
            if (pctChanged && pct <= 100) {
              lastReportedPct = pct;
            }
            if (bytesChanged) {
              lastReportedBytes = downloaded!;
            }
            if (pctChanged || bytesChanged) {
              const mb = downloaded != null ? (downloaded / 1024 / 1024).toFixed(1) : 'n/a';
              console.log(`[py:progress] ${pct}% (${mb}MB)`);
              onProgress?.({
                model: '',
                status: 'downloading',
                progress: pct,
                downloaded: downloaded ?? undefined,
              });
            }
          } else if (msg.type === 'done') {
            console.log(`[py:done] File saved to: ${msg.path}`);
            downloadedPath = msg.path || null;
          } else if (msg.type === 'error') {
            console.error(`[py:error] ${msg.message}`);
            reject(new Error(msg.message));
            return;
          } else if (msg.type === 'start') {
            console.log(`[py:start] Total size: ${msg.total} bytes`);
          }
        } catch {
          // Not a JSON line, ignore
        }
      }
    });

    proc.stderr.on('data', (data) => {
      const msg = data.toString().trim();
      if (msg) console.log(`[py:stderr] ${msg}`);
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      console.log(`[py:close] Process exited with code=${code}`);

      if (code === 0) {
        console.log(`[py:close] Success — resolving`);
        try { fs.unlinkSync(scriptFile); fs.rmdirSync(tmpDir); } catch { /* ignore */ }
        resolve(downloadedPath);
      } else {
        const errMsg = stderr || `Python download exited with code ${code}`;
        console.error(`[py:close] FAILURE — ${errMsg.slice(0, 300)}`);
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
