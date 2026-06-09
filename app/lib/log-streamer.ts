/**
 * LogStreamer — captures stdout/stderr from child processes
 * and makes logs available line-by-line with search support.
 */
export class LogStreamer {
  private lines: string[] = [];
  private partialLine = '';
  private listeners: Array<(line: string) => void> = [];
  private destroyed = false;

  /**
   * Register a callback for each new line.
   */
  onLine(callback: (line: string) => void): void {
    this.listeners.push(callback);
  }

  /**
   * Add raw data (e.g., from stdout/stderr 'data' events).
   */
  addData(data: string): void {
    if (this.destroyed) return;

    const input = this.partialLine + data;
    const parts = input.split('\n');

    // Last element may be incomplete (no trailing newline)
    this.partialLine = parts.pop() || '';

    for (const line of parts) {
      this.lines.push(line);
      for (const listener of this.listeners) {
        listener(line);
      }
    }
  }

  /**
   * Get all captured log lines.
   */
  getLines(): string[] {
    return [...this.lines];
  }

  /**
   * Search log lines by keyword (case-insensitive substring match).
   */
  search(keyword: string): string[] {
    const lower = keyword.toLowerCase();
    return this.lines.filter((line) => line.toLowerCase().includes(lower));
  }

  /**
   * Clear all stored lines.
   */
  clear(): void {
    this.lines = [];
    this.partialLine = '';
  }

  /**
   * Destroy the streamer and remove all listeners.
   */
  destroy(): void {
    this.destroyed = true;
    this.listeners = [];
    this.clear();
  }
}
