import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.resolve(process.cwd());
const PRESETS_DIR = path.join(PROJECT_ROOT, 'presets');
const PRESETS_FILE = path.join(PRESETS_DIR, 'presets.json');

/**
 * PresetStore — save and load parameter configurations as JSON presets.
 */
export class PresetStore {
  private loadAll(): Record<string, any> {
    if (!fs.existsSync(PRESETS_FILE)) {
      return {};
    }
    try {
      return JSON.parse(fs.readFileSync(PRESETS_FILE, 'utf-8'));
    } catch {
      return {};
    }
  }

  private saveAll(data: Record<string, any>): void {
    if (!fs.existsSync(PRESETS_DIR)) {
      fs.mkdirSync(PRESETS_DIR, { recursive: true });
    }
    fs.writeFileSync(PRESETS_FILE, JSON.stringify(data, null, 2));
  }

  /**
   * Save a parameter preset with the given name.
   */
  savePreset(name: string, params: Record<string, any>): void {
    const data = this.loadAll();
    data[name] = params;
    this.saveAll(data);
  }

  /**
   * Load a parameter preset by name.
   * Returns null if the preset does not exist.
   */
  loadPreset(name: string): Record<string, any> | null {
    const data = this.loadAll();
    return data[name] || null;
  }

  /**
   * List all preset names.
   */
  listPresets(): string[] {
    const data = this.loadAll();
    return Object.keys(data);
  }

  /**
   * Delete a preset by name.
   */
  deletePreset(name: string): void {
    const data = this.loadAll();
    delete data[name];
    this.saveAll(data);
  }
}
