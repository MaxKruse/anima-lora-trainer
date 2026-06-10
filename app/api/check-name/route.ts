import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.resolve(process.cwd());
const CONFIG_DIR = path.join(PROJECT_ROOT, '.config');
const CONFIG_FILE = path.join(CONFIG_DIR, 'app-config.json');

function loadConfig(): { outputDir: string } {
  try {
    if (!fs.existsSync(CONFIG_FILE)) {
      return { outputDir: path.join(PROJECT_ROOT, 'output') };
    }
    const raw = fs.readFileSync(CONFIG_FILE, 'utf8');
    const saved = JSON.parse(raw);
    return {
      outputDir: saved.outputDir || path.join(PROJECT_ROOT, 'output'),
    };
  } catch {
    return { outputDir: path.join(PROJECT_ROOT, 'output') };
  }
}

/**
 * GET /api/check-name?name=foo
 *
 * Checks if a training name (loraName) already exists in the output folder.
 * Returns { available: boolean }.
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const name = searchParams.get('name');

    if (!name) {
      return NextResponse.json(
        { error: 'Missing name parameter' },
        { status: 400 }
      );
    }

    const outputDir = loadConfig().outputDir;
    if (!fs.existsSync(outputDir)) {
      return NextResponse.json({ available: true, name });
    }

    const entries = fs.readdirSync(outputDir);
    const exists = entries.includes(name);

    return NextResponse.json({
      available: !exists,
      name,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to check name' },
      { status: 500 }
    );
  }
}
