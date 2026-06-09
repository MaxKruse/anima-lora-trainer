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
      className={`lora-download-link ${!exists ? 'disabled' : ''}`}
      aria-disabled={!exists}
    >
      {exists ? `Download ${loraFile}` : `${loraFile} (not available)`}
    </a>
  );
}
