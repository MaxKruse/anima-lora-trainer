import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.resolve(process.cwd());

/**
 * Create a zip archive of training images and captions.
 *
 * Scans the source directory for image files and their .txt caption files,
 * then creates a zip archive in the output directory.
 *
 * @param sourceDir - Directory containing training images and captions
 * @param outputDir - Directory where the zip file will be created
 * @returns Path to the created zip file, or null if no images found
 */
export async function createTrainingZip(
  sourceDir: string,
  outputDir: string
): Promise<string | null> {
  // Validate source directory exists
  if (!fs.existsSync(sourceDir)) {
    throw new Error(`Source directory not found: ${sourceDir}`);
  }

  // Check for image files
  const files = fs.readdirSync(sourceDir);
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'];
  const hasImages = files.some((f) =>
    imageExtensions.includes(path.extname(f).toLowerCase())
  );

  if (!hasImages) {
    console.log(`No image files found in ${sourceDir}, skipping zip creation`);
    return null;
  }

  const zipPath = path.join(outputDir, 'training-data.zip');

  return new Promise((resolve, reject) => {
    const scriptPath = path.join(PROJECT_ROOT, 'scripts', 'zip_training_data.py');

    const proc = spawn('uv', ['run', 'python', scriptPath, sourceDir, outputDir], {
      shell: true,
      cwd: PROJECT_ROOT,
    });

    let stderr = '';
    proc.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    proc.stdout.on('data', (data: Buffer) => {
      console.log(`[zip] ${data.toString().trim()}`);
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve(zipPath);
      } else if (code === 1 && stderr.includes('No image files')) {
        resolve(null);
      } else {
        reject(new Error(`Zip creation failed with code ${code}: ${stderr}`));
      }
    });

    proc.on('error', (err) => {
      reject(err);
    });
  });
}
