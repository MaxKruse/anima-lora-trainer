import { NextResponse } from 'next/server';
import { getResolvedModelManifest } from '../../lib/model-manifest';
import { checkLocalModel, downloadModel, DownloadProgress } from '../../lib/model-downloader';

interface ActiveDownload {
  controller: AbortController;
  progress: number;
  downloaded?: number;
  error?: string;
}

/** Active downloads tracked by model name → download state */
const activeDownloads = new Map<string, ActiveDownload>();

/**
 * GET /api/models
 *
 * Returns the download status of all Anima models.
 */
export async function GET() {
  try {
    const manifest = await getResolvedModelManifest('anima');

    const models = await Promise.all(
      manifest.map(async (entry) => {
        const status = checkLocalModel(entry);
        const active = activeDownloads.get(entry.name);
        const isActive = !!active;

        // Determine status — check in-memory state first, then local file state
        let modelStatus: 'pending' | 'downloading' | 'downloaded';
        if (isActive) {
          modelStatus = 'downloading';
        } else if (status.exists && status.downloadPercent === 100) {
          modelStatus = 'downloaded';
        } else if (status.exists && status.downloadPercent! > 0) {
          // Partial file (e.g., after server restart mid-download)
          modelStatus = 'downloading';
        } else {
          modelStatus = 'pending';
        }

        // Use in-memory progress from the onProgress callback when available,
        // otherwise fall back to file-size-based progress from local file
        const progress = isActive ? active.progress : (status.downloadPercent || 0);

        return {
          name: entry.name,
          hfRepo: entry.hfRepo,
          hfFile: entry.hfFile,
          expectedSizeBytes: entry.expectedSizeBytes,
          status: modelStatus,
          progress,
          downloaded: isActive ? active.downloaded : undefined,
          sizeBytes: status.sizeBytes,
          canAbort: isActive,
          error: active?.error,
          localPath: status.localPath,
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
 * Triggers download of a specific model (runs in background).
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

    const manifest = await getResolvedModelManifest('anima');
    const entry = manifest.find((e) => e.name === modelName);

    if (!entry) {
      return NextResponse.json(
        { error: `Unknown model: ${modelName}` },
        { status: 404 }
      );
    }

    // Check if already downloaded
    const status = checkLocalModel(entry);

    if (status.exists && status.downloadPercent === 100) {
      return NextResponse.json(
        { error: `Model ${modelName} is already downloaded`, status: 'already_downloaded' },
        { status: 409 }
      );
    }

    // Check if already downloading
    if (activeDownloads.has(modelName)) {
      return NextResponse.json(
        { error: `Model ${modelName} is already downloading`, status: 'already_downloading' },
        { status: 409 }
      );
    }

    // Create abort controller and start download in background
    const controller = new AbortController();
    activeDownloads.set(modelName, {
      controller,
      progress: 0,
    });

    // Fire-and-forget: run download in background
    (async () => {
      try {
        await downloadModel(
          entry,
          (progress: DownloadProgress) => {
            const dlEntry = activeDownloads.get(modelName);
            if (!dlEntry) return;

            if (progress.progress !== undefined) {
              dlEntry.progress = progress.progress;
            }
            if (progress.downloaded !== undefined) {
              dlEntry.downloaded = progress.downloaded;
            }
            if (progress.status === 'failed' && progress.error) {
              dlEntry.error = progress.error;
            }
          },
          controller.signal
        );
      } catch (err: any) {
        console.error(`[models] Download failed for ${modelName}: ${err.message || err}`);
      } finally {
        activeDownloads.delete(modelName);
      }
    })();

    return NextResponse.json({
      name: modelName,
      status: 'started',
      message: `Download initiated for ${modelName}`,
    });
  } catch (error: any) {
    console.error(`[models:POST] Unhandled error: ${error.message || error}`, error.stack);
    return NextResponse.json(
      { error: error.message || 'Failed to process request' },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/models?modelName=xxx
 *
 * Aborts an in-progress download.
 */
export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const modelName = searchParams.get('modelName');

    if (!modelName) {
      return NextResponse.json(
        { error: 'modelName query parameter is required' },
        { status: 400 }
      );
    }

    const active = activeDownloads.get(modelName);
    if (!active) {
      return NextResponse.json(
        { error: `No active download for ${modelName}` },
        { status: 404 }
      );
    }

    active.controller.abort();
    activeDownloads.delete(modelName);

    return NextResponse.json({
      name: modelName,
      status: 'aborted',
      message: `Download aborted for ${modelName}`,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to abort download' },
      { status: 500 }
    );
  }
}
