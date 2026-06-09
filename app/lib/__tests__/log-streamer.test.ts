import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

async function importLogStreamer() {
  const mod = await import('../log-streamer');
  return mod.LogStreamer;
}

describe('LogStreamer', () => {
  let streamer: any;

  afterEach(() => {
    if (streamer) {
      streamer.destroy();
    }
  });

  it('captures stdout/stderr from child process', async () => {
    const LogStreamer = await importLogStreamer();

    const lines: string[] = [];
    streamer = new LogStreamer();
    streamer.onLine((line: string) => lines.push(line));

    // Simulate receiving data
    streamer.addData('Line 1\nLine 2\n');
    streamer.addData('Line 3\n');

    expect(lines).toEqual(['Line 1', 'Line 2', 'Line 3']);
  });

  it('makes logs available line-by-line', async () => {
    const LogStreamer = await importLogStreamer();

    streamer = new LogStreamer();

    streamer.addData('First line\n');
    streamer.addData('Second line\n');

    const allLines = streamer.getLines();
    expect(allLines).toHaveLength(2);
    expect(allLines[0]).toBe('First line');
    expect(allLines[1]).toBe('Second line');
  });

  it('supports searching/filtering log lines', async () => {
    const LogStreamer = await importLogStreamer();

    streamer = new LogStreamer();
    streamer.addData('INFO: Training started\n');
    streamer.addData('ERROR: Out of memory\n');
    streamer.addData('INFO: Epoch 1/10\n');
    streamer.addData('WARNING: Low VRAM\n');

    const infoLines = streamer.search('INFO');
    expect(infoLines).toHaveLength(2);
    expect(infoLines[0]).toContain('Training started');
    expect(infoLines[1]).toContain('Epoch 1/10');

    const errorLines = streamer.search('ERROR');
    expect(errorLines).toHaveLength(1);
    expect(errorLines[0]).toContain('Out of memory');
  });
});
