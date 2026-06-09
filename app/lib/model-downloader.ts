import { spawn } from 'child_process';
import { ModelEntry } from './model-manifest';
import fs from 'fs';
import path from 'path';

export interface DownloadProgress {
  model: string;
  status: 'started' | 'downloading' | 'completed' | 'failed';
  progress?: number;    // 0-100
  message?: string;
  error?: string;
}

const MAX_RETRIES = 3;

export async function downloadModel(
  entry: ModelEntry,
  onProgress?: (progress: DownloadProgress) => void,
  abortSignal?: AbortSignal
): Promise<void> {
  const { hfRepo, hfFile, localPath } = entry;
  const localDir = path.dirname(localPath);

  // Ensure target directory exists
  if (!fs.existsSync(localDir)) {
    fs.mkdirSync(localDir, { recursive: true });
  }

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    if (abortSignal?.aborted) {
      throw new Error(`Download aborted: ${entry.name}`);
    }

    onProgress?.({
      model: entry.name,
      status: attempt > 1 ? 'downloading' : 'started',
      message: `Downloading ${entry.name} (attempt ${attempt}/${MAX_RETRIES})...`,
    });

    try {
      await runHuggingfaceDownload(hfRepo, hfFile, localDir, abortSignal);

      onProgress?.({
        model: entry.name,
        status: 'completed',
        progress: 100,
        message: `Downloaded ${entry.name} successfully`,
      });

      return;
    } catch (error: any) {
      const errorMsg = error.message || 'Unknown error';

      if (attempt === MAX_RETRIES) {
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
  localDir: string,
  signal?: AbortSignal
): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(
      'huggingface-cli',
      ['download', repo, file, '--local-dir', localDir],
      { shell: true }
    );

    let stderr = '';

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(stderr || `huggingface-cli exited with code ${code}`));
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });

    if (signal) {
      signal.addEventListener('abort', () => {
        proc.kill();
        reject(new Error('Download aborted'));
      });
    }
  });
}

export async function checkModelStatus(entry: ModelEntry): Promise<{
  exists: boolean;
  sizeBytes?: number;
  downloadPercent?: number;
}> {
  const fullPath = path.resolve(entry.localPath);

  if (!fs.existsSync(fullPath)) {
    return { exists: false };
  }

  const stat = fs.statSync(fullPath);
  const percent = Math.round((stat.size / entry.expectedSizeBytes) * 100);

  return {
    exists: true,
    sizeBytes: stat.size,
    downloadPercent: Math.min(percent, 100),
  };
}
