import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.resolve(process.cwd());
const CONFIG_DIR = path.join(PROJECT_ROOT, '.config');
const CONFIG_FILE = path.join(CONFIG_DIR, 'app-config.json');

interface AppConfig {
  trainingImagesDir: string;
  outputDir: string;
  modelsDir?: string;
  sdCliPath?: string;
  sdScriptsPath?: string;
}

const DEFAULT_CONFIG: AppConfig = {
  trainingImagesDir: '',
  outputDir: path.join(PROJECT_ROOT, 'outputs'),
  modelsDir: path.join(PROJECT_ROOT, 'models'),
  sdCliPath: '',
  sdScriptsPath: path.join(PROJECT_ROOT, 'sd-scripts'),
};

function loadConfig(): AppConfig {
  try {
    if (!fs.existsSync(CONFIG_FILE)) {
      return { ...DEFAULT_CONFIG };
    }
    const raw = fs.readFileSync(CONFIG_FILE, 'utf-8');
    const saved = JSON.parse(raw) as Partial<AppConfig>;
    return { ...DEFAULT_CONFIG, ...saved };
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

function saveConfig(config: AppConfig): void {
  try {
    if (!fs.existsSync(CONFIG_DIR)) {
      fs.mkdirSync(CONFIG_DIR, { recursive: true });
    }
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
  } catch (error: any) {
    console.error('Failed to save config:', error.message);
  }
}

/**
 * GET /api/config
 * Return current app configuration.
 */
export async function GET() {
  try {
    const config = loadConfig();
    return NextResponse.json({ config });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to load config' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/config
 * Save app configuration.
 *
 * Body: { config: Partial<AppConfig> }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const updates = body.config as Partial<AppConfig>;

    if (!updates || typeof updates !== 'object') {
      return NextResponse.json(
        { error: 'Invalid config object' },
        { status: 400 }
      );
    }

    const current = loadConfig();
    const merged = { ...current, ...updates };
    saveConfig(merged);

    return NextResponse.json({ config: merged });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to save config' },
      { status: 500 }
    );
  }
}
