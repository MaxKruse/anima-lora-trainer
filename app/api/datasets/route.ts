import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const DATASETS_DIR = path.resolve(process.cwd(), 'datasets');

/**
 * GET /api/datasets
 * List subdirectories inside the project's datasets/ folder.
 * Returns: { directories: Array<{ name: string; path: string }> }
 */
export async function GET() {
  try {
    if (!fs.existsSync(DATASETS_DIR)) {
      return NextResponse.json({
        directories: [],
        datasetsDir: DATASETS_DIR,
        exists: false,
      });
    }

    const entries = fs.readdirSync(DATASETS_DIR, { withFileTypes: true });
    const directories = entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => ({
        name: entry.name,
        path: path.join(DATASETS_DIR, entry.name),
      }));

    return NextResponse.json({
      directories,
      datasetsDir: DATASETS_DIR,
      exists: true,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to list datasets' },
      { status: 500 }
    );
  }
}
