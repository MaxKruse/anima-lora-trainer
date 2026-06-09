import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

/**
 * POST /api/config/verify
 * Check if a given path exists and is a directory.
 *
 * Body: { path: string }
 * Returns: { exists: boolean, isDirectory: boolean, path: string }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const targetPath = body.path as string;

    if (!targetPath || typeof targetPath !== 'string') {
      return NextResponse.json(
        { error: 'A path string is required' },
        { status: 400 }
      );
    }

    // Resolve relative paths against project root
    const resolved = path.isAbsolute(targetPath)
      ? targetPath
      : path.resolve(process.cwd(), targetPath);

    let exists = false;
    let isDirectory = false;

    try {
      const stat = fs.statSync(resolved);
      exists = true;
      isDirectory = stat.isDirectory();
    } catch {
      // Path doesn't exist or can't be accessed
    }

    return NextResponse.json({
      exists,
      isDirectory,
      path: resolved,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Verification failed' },
      { status: 500 }
    );
  }
}
