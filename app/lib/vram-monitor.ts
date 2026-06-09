/**
 * GPU VRAM monitoring — parse nvidia-smi JSON output.
 */

export interface VramInfo {
  used: number;
  total: number;
  percentage: number;
}

/**
 * Parse nvidia-smi JSON output and return VRAM usage info.
 *
 * Expects the output of `nvidia-smi --query-gpu=name,fb_memory_usage,bar1_memory_usage --format=csv`.
 * Returns { used: 0, total: 0, percentage: 0 } on parse errors.
 */
export function parseVramUsage(output: string): VramInfo {
  try {
    if (!output.trim()) {
      return { used: 0, total: 0, percentage: 0 };
    }

    const data = JSON.parse(output);
    const gpuList = data.attachments?.nvidia_smi?.gpu;

    if (!gpuList || !Array.isArray(gpuList) || gpuList.length === 0) {
      return { used: 0, total: 0, percentage: 0 };
    }

    // Use the first GPU
    const gpu = gpuList[0];
    const memStr = gpu.fb_memory_usage || '';

    // Parse "XXXX MiB / YYYY MiB"
    const match = memStr.match(/(\d+)\s*MiB\s*\/\s*(\d+)\s*MiB/);

    if (!match) {
      return { used: 0, total: 0, percentage: 0 };
    }

    const used = parseInt(match[1], 10);
    const total = parseInt(match[2], 10);
    const percentage = total > 0 ? (used / total) * 100 : 0;

    return { used, total, percentage: Math.round(percentage * 10) / 10 };
  } catch {
    return { used: 0, total: 0, percentage: 0 };
  }
}
