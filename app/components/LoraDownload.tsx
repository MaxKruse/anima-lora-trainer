'use client';

interface LoraDownloadProps {
  runId: string;
  loraFile: string;
  exists: boolean;
}

/**
 * Download link for a LoRA .safetensors file.
 */
export function LoraDownload({ runId, loraFile, exists }: LoraDownloadProps) {
  const downloadUrl = `/api/download?runId=${encodeURIComponent(runId)}&file=${encodeURIComponent(loraFile)}`;

  return (
    <a
      href={downloadUrl}
      download={loraFile}
      className={`text-sm font-medium transition-colors ${
        exists
          ? 'text-slate-900 dark:text-slate-100 hover:underline'
          : 'text-slate-400 dark:text-slate-500 cursor-not-allowed'
      }`}
      aria-disabled={!exists}
    >
      {exists ? `Download ${loraFile}` : `${loraFile} (not available)`}
    </a>
  );
}
