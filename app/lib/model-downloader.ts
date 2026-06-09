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
  const fullPath = path.resolve(localPath);

  console.log(`[download] === downloadModel START: ${entry.name} ===`);
  console.log(`[download]   hfRepo=${hfRepo}`);
  console.log(`[download]   hfFile=${hfFile}`);
  console.log(`[download]   localPath=${localPath}`);
  console.log(`[download]   localDir=${localDir}`);
  console.log(`[download]   fullPath (resolved)=${fullPath}`);

  // Ensure target directory exists
  if (!fs.existsSync(localDir)) {
    console.log(`[download] Creating directory: ${localDir}`);
    fs.mkdirSync(localDir, { recursive: true });
  } else {
    console.log(`[download] Directory exists: ${localDir}`);
  }

  // hf download preserves repo dir structure — file lands at <localDir>/<hfFile>
  const hfActualPath = path.join(localDir, hfFile);
  const resolvedActual = path.resolve(hfActualPath);

  // Check if file already exists before starting
  if (fs.existsSync(fullPath)) {
    const existingStat = fs.statSync(fullPath);
    console.log(`[download] WARNING: File already exists at ${fullPath}, size=${existingStat.size} bytes`);
  } else if (fs.existsSync(resolvedActual)) {
    const existingStat = fs.statSync(resolvedActual);
    console.log(`[download] WARNING: File already exists at nested path ${resolvedActual}, size=${existingStat.size} bytes`);
  } else {
    console.log(`[download] File does not exist yet at ${fullPath}`);
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
      console.log(`[download] Spawning hf download: hf download ${hfRepo} ${hfFile} --local-dir ${localDir}`);

      // Disk-based progress fallback: poll nested file size while hf runs
      let diskLastProgress = -1;
      const diskInterval = setInterval(() => {
        for (const checkPath of [resolvedActual, fullPath]) {
          try {
            if (fs.existsSync(checkPath)) {
              const s = fs.statSync(checkPath);
              if (entry.expectedSizeBytes > 0) {
                const pct = Math.round((s.size / entry.expectedSizeBytes) * 100);
                if (pct !== diskLastProgress && pct <= 100) {
                  diskLastProgress = pct;
                  console.log(`[download:disk] ${checkPath}: ${s.size} / ${entry.expectedSizeBytes} = ${pct}%`);
                  onProgress?.({
                    model: entry.name,
                    status: 'downloading',
                    progress: pct,
                  });
                }
              }
              break;
            }
          } catch {
            // File might be in use — skip
          }
        }
      }, 1000);

      try {
        await runHuggingfaceDownload(hfRepo, hfFile, localDir, abortSignal, (progress) => {
          onProgress?.({ ...progress, model: entry.name });
        });
      } finally {
        clearInterval(diskInterval);
      }

      console.log(`[download] hf process exited successfully for ${entry.name}`);

      if (fs.existsSync(fullPath)) {
        // File already at expected location (e.g., hf --local-dir put it flat)
        const downloadedStat = fs.statSync(fullPath);
        console.log(`[download] File already at expected path: ${fullPath}, size=${downloadedStat.size} bytes`);
      } else if (fs.existsSync(resolvedActual)) {
        // hf preserved repo structure — move file to flat location
        console.log(`[download] hf placed file at nested path: ${resolvedActual}`);
        console.log(`[download] Moving to expected path: ${fullPath}`);

        const actualStat = fs.statSync(resolvedActual);
        console.log(`[download] Source file size: ${actualStat.size} bytes`);

        // Ensure target directory exists
        const targetDir = path.dirname(fullPath);
        if (!fs.existsSync(targetDir)) {
          fs.mkdirSync(targetDir, { recursive: true });
        }

        fs.renameSync(resolvedActual, fullPath);
        console.log(`[download] Moved file successfully`);

        // Clean up empty intermediate directories left by hf
        let cleanupDir = path.dirname(resolvedActual);
        while (cleanupDir && cleanupDir !== localDir && cleanupDir !== path.parse(cleanupDir).root) {
          try {
            const entries = fs.readdirSync(cleanupDir);
            if (entries.length === 0) {
              console.log(`[download] Removing empty dir: ${cleanupDir}`);
              fs.rmdirSync(cleanupDir);
            } else {
              break;
            }
          } catch {
            break;
          }
          cleanupDir = path.dirname(cleanupDir);
        }

        // Verify the move
        const movedStat = fs.statSync(fullPath);
        console.log(`[download] Verified moved file at ${fullPath}, size=${movedStat.size} bytes, expected=${entry.expectedSizeBytes} bytes`);
      } else {
        console.error(`[download] ERROR: hf exited 0 but file MISSING at both locations!`);
        console.error(`[download]   Expected: ${fullPath}`);
        console.error(`[download]   Nested:   ${resolvedActual}`);
      }

      // List directory contents for debugging
      try {
        const dirContents = fs.readdirSync(localDir);
        console.log(`[download] Contents of ${localDir}: ${dirContents.join(', ') || '(empty)'}`);
      } catch (readErr: any) {
        console.error(`[download] Failed to read directory ${localDir}: ${readErr.message}`);
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

      // Check disk state after failure
      if (fs.existsSync(fullPath)) {
        const partialStat = fs.statSync(fullPath);
        console.error(`[download] Partial file exists at ${fullPath}, size=${partialStat.size} bytes`);
      } else if (fs.existsSync(resolvedActual)) {
        const partialStat = fs.statSync(resolvedActual);
        console.error(`[download] Partial file exists at nested path ${resolvedActual}, size=${partialStat.size} bytes`);
      } else {
        console.error(`[download] No file found after failure (checked ${fullPath} and ${resolvedActual})`);
      }

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
  localDir: string,
  signal?: AbortSignal,
  onProgress?: (progress: DownloadProgress) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log(`[hf:spawn] Spawning: hf download ${repo} ${file} --local-dir ${localDir}`);

    // Force UTF-8 encoding for Python's stdout/stderr — fixes Windows cp1252
    // "charmap codec can't encode character '\u2713'" crash from hf CLI progress output
    const spawnEnv = { ...process.env, PYTHONIOENCODING: 'utf-8' };

    const proc = spawn(
      'hf',
      ['download', repo, file, '--local-dir', localDir],
      { shell: false, env: spawnEnv }
    );

    console.log(`[hf:spawn] Process spawned, pid=${proc.pid}`);

    let stderr = '';
    let stdout = '';
    let lastReportedProgress = -1;

    // Accumulate raw bytes then parse — handles \r inline progress, partial chunks, ANSI codes
    let rawStdout = '';
    let rawStderr = '';

    proc.stdout.on('data', (data) => {
      const chunk = data.toString();
      rawStdout += chunk;
      stdout += chunk;
    });

    proc.stderr.on('data', (data) => {
      const chunk = data.toString();
      rawStderr += chunk;
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
      if (stderr) {
        console.log(`[hf:close] Full stderr (${stderr.length} chars): ${stderr.replace(/\n/g, ' | ')}`);
      }
      if (stdout) {
        console.log(`[hf:close] Full stdout (${stdout.length} chars): ${stdout.replace(/\n/g, ' | ')}`);
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

export async function checkModelStatus(entry: ModelEntry): Promise<{
  exists: boolean;
  sizeBytes?: number;
  downloadPercent?: number;
}> {
  const fullPath = path.resolve(entry.localPath);
  const localDir = path.dirname(fullPath);
  const dirPath = path.dirname(fullPath);

  // hf download preserves repo structure — also check nested path
  const nestedPath = path.resolve(path.join(localDir, entry.hfFile));

  console.log(`[status] checkModelStatus: ${entry.name}`);
  console.log(`[status]   localPath (raw)=${entry.localPath}`);
  console.log(`[status]   fullPath (resolved)=${fullPath}`);
  console.log(`[status]   nestedPath (hf actual)=${nestedPath}`);

  // Check flat path first
  if (fs.existsSync(fullPath)) {
    const stat = fs.statSync(fullPath);
    const percent = entry.expectedSizeBytes > 0
      ? Math.round((stat.size / entry.expectedSizeBytes) * 100)
      : 0;
    console.log(`[status]   File EXISTS at flat path: size=${stat.size} bytes, expected=${entry.expectedSizeBytes} bytes, percent=${Math.min(percent, 100)}%`);
    return {
      exists: true,
      sizeBytes: stat.size,
      downloadPercent: Math.min(percent, 100),
    };
  }

  // Check nested path (hf may have placed file here during download)
  if (fs.existsSync(nestedPath)) {
    const stat = fs.statSync(nestedPath);
    const percent = entry.expectedSizeBytes > 0
      ? Math.round((stat.size / entry.expectedSizeBytes) * 100)
      : 0;
    console.log(`[status]   File EXISTS at nested path: size=${stat.size} bytes, expected=${entry.expectedSizeBytes} bytes, percent=${Math.min(percent, 100)}%`);
    return {
      exists: true,
      sizeBytes: stat.size,
      downloadPercent: Math.min(percent, 100),
    };
  }

  // Not found anywhere
  if (fs.existsSync(dirPath)) {
    const dirContents = fs.readdirSync(dirPath);
    console.log(`[status]   File NOT found. Directory ${dirPath} exists, contents: ${dirContents.join(', ') || '(empty)'}`);
  } else {
    console.log(`[status]   File NOT found. Directory ${dirPath} does NOT exist.`);
  }
  return { exists: false };
}
