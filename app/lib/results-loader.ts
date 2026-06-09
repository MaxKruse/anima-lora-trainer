import fs from 'fs';
import path from 'path';

/**
 * Combined result entry with both training params and evaluation data.
 */
export interface ResultEntry {
  params: Record<string, any>;
  loraFile: string | null;
  imageFile: string | null;
  status: string;
  error: string | null;
  inferenceTimeMs: number | null;
}

/**
 * Parse manifest.json + evaluation.json into structured result objects.
 *
 * Reads both files from the run directory and merges permutation params
 * with evaluation results by matching perm_name.
 */
export function loadResults(runDir: string): ResultEntry[] {
  const manifestPath = path.join(runDir, 'manifest.json');
  const evalPath = path.join(runDir, 'evaluation.json');

  // Load manifest
  const manifestData = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  const permutations = manifestData.permutations || [];

  // Load evaluation.json if it exists
  let evalResults: any[] = [];
  if (fs.existsSync(evalPath)) {
    const evalData = JSON.parse(fs.readFileSync(evalPath, 'utf-8'));
    evalResults = evalData.results || [];
  }

  // Build a lookup map from perm_name to evaluation result
  const evalMap = new Map<string, any>();
  for (const evalEntry of evalResults) {
    evalMap.set(evalEntry.perm_name, evalEntry);
  }

  // Merge permutation params with evaluation results
  return permutations.map((perm: any) => {
    // Find matching evaluation result by perm_name
    // The perm_name is derived from the permutation's output folder
    const permName = perm.params?._folderName || `perm-${perm.index}`;
    const evalEntry = evalMap.get(permName);

    // Also try matching by output_files
    let matchedEval = evalEntry;
    if (!matchedEval) {
      // Try to find by lora_file match
      for (const [name, entry] of evalMap) {
        if (
          perm.output_files &&
          perm.output_files.includes(entry.lora_file)
        ) {
          matchedEval = entry;
          break;
        }
      }
    }

    return {
      params: perm.params || {},
      loraFile: matchedEval?.lora_file ?? perm.output_files?.[0] ?? null,
      imageFile: matchedEval?.image_file ?? null,
      status: matchedEval?.status ?? perm.status,
      error: matchedEval?.error ?? perm.error ?? null,
      inferenceTimeMs: matchedEval?.inference_time_ms ?? null,
    };
  });
}
