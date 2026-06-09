import { NextResponse } from 'next/server';
import { getResolvedModelManifest } from '../../lib/model-manifest';
import { checkModelStatus, downloadModel, DownloadProgress } from '../../lib/model-downloader';

interface ActiveDownload {
  controller: AbortController;
  progress: number;
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
    console.log('[models:GET] Fetching model status for all models');
    const manifest = await getResolvedModelManifest('anima');
    console.log(`[models:GET] Manifest has ${manifest.length} models`);

    const models = await Promise.all(
      manifest.map(async (entry) => {
        const status = await checkModelStatus(entry);
        const active = activeDownloads.get(entry.name);
        const isActive = !!active;

        // Determine status — check in-memory state first, then disk state
        let modelStatus: 'pending' | 'downloading' | 'downloaded';
        if (isActive) {
          modelStatus = 'downloading';
        } else if (status.exists && status.downloadPercent === 100) {
          modelStatus = 'downloaded';
        } else if (status.exists && status.downloadPercent! > 0) {
          // Partial file on disk (e.g., after server restart mid-download)
          modelStatus = 'downloading';
        } else {
          modelStatus = 'pending';
        }

        // Use in-memory progress from the onProgress callback when available,
        // otherwise fall back to file-size-based progress from disk
        const progress = isActive ? active.progress : (status.downloadPercent || 0);

        console.log(`[models:GET] ${entry.name}: status=${modelStatus}, progress=${progress}%, exists=${status.exists}, sizeBytes=${status.sizeBytes}, expectedSizeBytes=${entry.expectedSizeBytes}, isActive=${isActive}, activeError=${active?.error ?? 'none'}`);

        return {
          name: entry.name,
          localPath: entry.localPath,
          expectedSizeBytes: entry.expectedSizeBytes,
          status: modelStatus,
          progress,
          sizeBytes: status.sizeBytes,
          canAbort: isActive,
          error: active?.error,
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

    console.log(`[models:POST] Received download request, modelName=${modelName}`);

    if (!modelName) {
      console.log('[models:POST] Rejected: modelName is missing');
      return NextResponse.json(
        { error: 'modelName is required' },
        { status: 400 }
      );
    }

    const manifest = await getResolvedModelManifest('anima');
    const entry = manifest.find((e) => e.name === modelName);

    if (!entry) {
      console.log(`[models:POST] Rejected: Unknown model "${modelName}". Available: ${manifest.map(e => e.name).join(', ')}`);
      return NextResponse.json(
        { error: `Unknown model: ${modelName}` },
        { status: 404 }
      );
    }

    console.log(`[models:POST] Found entry: hfRepo=${entry.hfRepo}, hfFile=${entry.hfFile}, localPath=${entry.localPath}`);

    // Check if already downloaded
    const status = await checkModelStatus(entry);
    console.log(`[models:POST] checkModelStatus: exists=${status.exists}, sizeBytes=${status.sizeBytes}, downloadPercent=${status.downloadPercent}`);

    if (status.exists && status.downloadPercent === 100) {
      console.log(`[models:POST] Rejected: ${modelName} already downloaded`);
      return NextResponse.json(
        { error: `Model ${modelName} is already downloaded`, status: 'already_downloaded' },
        { status: 409 }
      );
    }

    // Check if already downloading
    if (activeDownloads.has(modelName)) {
      console.log(`[models:POST] Rejected: ${modelName} already downloading`);
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
    console.log(`[models:POST] Starting background download for ${modelName}`);

    // Fire-and-forget: run download in background
    (async () => {
      try {
        console.log(`[models:POST:bg] Calling downloadModel for ${modelName}`);
        await downloadModel(
          entry,
          (progress: DownloadProgress) => {
            const dlEntry = activeDownloads.get(modelName);
            if (!dlEntry) {
              console.log(`[models:POST:bg] Progress callback dropped — no active entry for ${modelName}`);
              return;
            }

            console.log(`[models:POST:bg] Progress update for ${modelName}: status=${progress.status}, progress=${progress.progress ?? 'n/a'}, message=${progress.message ?? 'n/a'}, error=${progress.error ?? 'n/a'}`);

            if (progress.progress !== undefined) {
              dlEntry.progress = progress.progress;
            }

            if (progress.status === 'failed' && progress.error) {
              console.log(`[models:POST:bg] Setting error for ${modelName}: ${progress.error}`);
              dlEntry.error = progress.error;
            }
          },
          controller.signal
        );
        console.log(`[models:POST:bg] downloadModel completed successfully for ${modelName}`);
      } catch (err: any) {
        console.error(`[models:POST:bg] downloadModel threw for ${modelName}: ${err.message || err}`);
        // Error captured via onProgress callback
      } finally {
        console.log(`[models:POST:bg] Cleaning up activeDownloads for ${modelName}`);
        activeDownloads.delete(modelName);
      }
    })();

    console.log(`[models:POST] Returning success for ${modelName}`);
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
