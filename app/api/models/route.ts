import { NextResponse } from 'next/server';
import { getModelManifest, ModelEntry } from '../../lib/model-manifest';
import { checkModelStatus, downloadModel } from '../../lib/model-downloader';

/**
 * GET /api/models
 *
 * Returns the download status of all Anima models.
 */
export async function GET() {
  try {
    const manifest = getModelManifest('anima');
    const models = await Promise.all(
      manifest.map(async (entry) => {
        const status = await checkModelStatus(entry);
        return {
          name: entry.name,
          localPath: entry.localPath,
          expectedSizeBytes: entry.expectedSizeBytes,
          status: status.exists && status.downloadPercent === 100
            ? 'downloaded'
            : status.exists && status.downloadPercent! > 0
              ? 'downloading'
              : 'pending',
          progress: status.downloadPercent || 0,
          sizeBytes: status.sizeBytes,
        };
      })
    );

    return NextResponse.json({ models });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to get model status' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/models
 *
 * Body: { modelName: string }
 * Triggers download of a specific model.
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { modelName } = body as { modelName?: string };

    if (!modelName) {
      return NextResponse.json(
        { error: 'modelName is required' },
        { status: 400 }
      );
    }

    const manifest = getModelManifest('anima');
    const entry = manifest.find((e) => e.name === modelName);

    if (!entry) {
      return NextResponse.json(
        { error: `Unknown model: ${modelName}` },
        { status: 404 }
      );
    }

    // Check if already downloaded
    const status = await checkModelStatus(entry);
    if (status.exists && status.downloadPercent === 100) {
      return NextResponse.json(
        { error: `Model ${modelName} is already downloaded`, status: 'already_downloaded' },
        { status: 409 }
      );
    }

    // Start download
    try {
      await downloadModel(entry);
      return NextResponse.json({
        name: modelName,
        status: 'started',
        message: `Download initiated for ${modelName}`,
      });
    } catch (downloadError: any) {
      return NextResponse.json(
        { error: downloadError.message || 'Download failed', status: 'failed' },
        { status: 422 }
      );
    }
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to process request' },
      { status: 500 }
    );
  }
}
