import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.resolve(process.cwd());
const OUTPUT_DIR = path.join(PROJECT_ROOT, 'output');

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

    if (!fs.existsSync(OUTPUT_DIR)) {
      return NextResponse.json({ available: true, name });
    }

    const entries = fs.readdirSync(OUTPUT_DIR);
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
