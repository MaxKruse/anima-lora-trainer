import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { loadResults } from '../../lib/results-loader';

const PROJECT_ROOT = path.resolve(process.cwd());
const CONFIG_DIR = path.join(PROJECT_ROOT, '.config');
const CONFIG_FILE = path.join(CONFIG_DIR, 'app-config.json');

function loadConfig(): { outputDir: string } {
  try {
    if (!fs.existsSync(CONFIG_FILE)) {
      return { outputDir: path.join(PROJECT_ROOT, 'output') };
    }
    const raw = fs.readFileSync(CONFIG_FILE, 'utf8');
    const saved = JSON.parse(raw);
    return {
      outputDir: saved.outputDir || path.join(PROJECT_ROOT, 'output'),
    };
  } catch {
    return { outputDir: path.join(PROJECT_ROOT, 'output') };
  }
}

function getOutputDir(): string {
  return loadConfig().outputDir;
}

/**
 * GET /api/results
 *
 * Browse results or fetch detailed results for a specific run.
 *
 * Query params:
 *   - runId: Return detailed results for this run
 *   - sort: Sort results by parameter name (e.g., ?sort=network-dim)
 *   - filter: Filter results by param:value (e.g., ?filter=network-dim:32)
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const runId = searchParams.get('runId');
    const sortBy = searchParams.get('sort');
    const filterParam = searchParams.get('filter');

    if (runId) {
      // Return detailed results for a specific run
      const runDir = path.join(getOutputDir(), runId);

      if (!fs.existsSync(runDir)) {
        return NextResponse.json(
          { error: `Run ${runId} not found` },
          { status: 404 }
        );
      }

      let results = loadResults(runDir);

      // Apply filter
      if (filterParam) {
        const [paramName, paramValue] = filterParam.split(':');
        const numericValue = parseFloat(paramValue);
        results = results.filter((r) => {
          const actual = r.params[paramName];
          if (!isNaN(numericValue) && typeof actual === 'number') {
            return actual === numericValue;
          }
          return String(actual) === paramValue;
        });
      }

      // Apply sort
      if (sortBy) {
        results.sort((a, b) => {
          const aVal = a.params[sortBy];
          const bVal = b.params[sortBy];
          if (typeof aVal === 'number' && typeof bVal === 'number') {
            return aVal - bVal;
          }
          return String(aVal).localeCompare(String(bVal));
        });
      }

      return NextResponse.json({
        runId,
        total: results.length,
        results,
      });
    }

    // Return list of all completed runs
    const runs = [];

    const outputDir = getOutputDir();
    if (fs.existsSync(outputDir)) {
      const entries = fs.readdirSync(outputDir);

      for (const entry of entries) {
        const runDir = path.join(outputDir, entry);
        const manifestPath = path.join(runDir, 'manifest.json');

        if (fs.existsSync(manifestPath)) {
          try {
            const manifest = JSON.parse(
              fs.readFileSync(manifestPath, 'utf-8')
            );
            runs.push({
              runId: entry,
              total: manifest.total || 0,
              completed: manifest.completed || 0,
              failed: manifest.failed || 0,
            });
          } catch {
            // Skip invalid manifest files
          }
        }
      }
    }

    return NextResponse.json({ runs });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch results' },
      { status: 500 }
    );
  }
}
