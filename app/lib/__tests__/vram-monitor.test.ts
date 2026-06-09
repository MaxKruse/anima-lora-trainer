import { describe, it, expect } from 'vitest';

async function importVramMonitor() {
  const mod = await import('../vram-monitor');
  return { parseVramUsage: mod.parseVramUsage };
}

describe('parseVramUsage', () => {
  it('parses nvidia-smi JSON output for VRAM used/total', async () => {
    const { parseVramUsage } = await importVramMonitor();

    const nvidiaSmiOutput = JSON.stringify({
      attachments: {
        nvidia_smi: {
          gpu: [
            {
              name: 'NVIDIA GeForce RTX 4090',
              fb_memory_usage: '8192 MiB / 24576 MiB',
              bar1_memory_usage: '2048 MiB / 16384 MiB',
            },
          ],
        },
      },
    });

    const result = parseVramUsage(nvidiaSmiOutput);

    expect(result.used).toBe(8192);
    expect(result.total).toBe(24576);
  });

  it('returns percentage used', async () => {
    const { parseVramUsage } = await importVramMonitor();

    const nvidiaSmiOutput = JSON.stringify({
      attachments: {
        nvidia_smi: {
          gpu: [
            {
              name: 'NVIDIA GeForce RTX 4090',
              fb_memory_usage: '12288 MiB / 24576 MiB',
            },
          ],
        },
      },
    });

    const result = parseVramUsage(nvidiaSmiOutput);

    expect(result.percentage).toBeCloseTo(50, 1);
  });

  it('handles nvidia-smi failure gracefully', async () => {
    const { parseVramUsage } = await importVramMonitor();

    // Empty/invalid output
    const result1 = parseVramUsage('');
    expect(result1.used).toBe(0);
    expect(result1.total).toBe(0);
    expect(result1.percentage).toBe(0);

    // Malformed JSON
    const result2 = parseVramUsage('not json at all');
    expect(result2.used).toBe(0);
    expect(result2.total).toBe(0);

    // Missing GPU data
    const result3 = parseVramUsage('{}');
    expect(result3.used).toBe(0);
    expect(result3.total).toBe(0);
  });
});
