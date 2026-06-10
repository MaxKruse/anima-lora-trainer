# Issue #5 — `LoraDownload` Links to Non-Existent `/api/download` Endpoint 🟡 HIGH

## What's Broken

The `LoraDownload` component renders download links pointing to `/api/download?runId=...&file=...`. **This API route does not exist.** Clicking the link returns a Next.js 404.

## Where

**File:** `app/components/LoraDownload.tsx`

```typescript
const downloadUrl = `/api/download?runId=${encodeURIComponent(runId)}&file=${encodeURIComponent(loraFile)}`;

return (
  <a href={downloadUrl} download={loraFile}>
    Download {loraFile}
  </a>
);
```

**Missing:** `app/api/download/route.ts`

## What Needs to Happen

### Option A: Create `/api/download` endpoint (recommended)

```typescript
// app/api/download/route.ts
import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.resolve(process.cwd());
const OUTPUT_DIR = path.join(PROJECT_ROOT, 'output');

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const runId = searchParams.get('runId');
  const file = searchParams.get('file');

  if (!runId || !file) {
    return NextResponse.json({ error: 'runId and file are required' }, { status: 400 });
  }

  // Prevent path traversal
  const safeFile = path.basename(file);
  const runDir = path.join(OUTPUT_DIR, path.basename(runId));

  // Search for the file in the run directory (may be in subdirectories)
  const filePath = findFileInDir(runDir, safeFile);
  if (!filePath || !fs.existsSync(filePath)) {
    return NextResponse.json({ error: 'File not found' }, { status: 404 });
  }

  const content = fs.readFileSync(filePath);
  return new Response(content, {
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Disposition': `attachment; filename="${safeFile}"`,
    },
  });
}

function findFileInDir(dir: string, fileName: string): string | null {
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const found = findFileInDir(fullPath, fileName);
        if (found) return found;
      } else if (entry.name === fileName) {
        return fullPath;
      }
    }
  } catch { /* ignore */ }
  return null;
}
```

### Option B: Serve directly from output/ directory

If Next.js static file serving can reach `output/`, use direct paths:

```typescript
// In LoraDownload.tsx
const downloadUrl = `/output/${runId}/${loraFile}`;
```

This requires configuring `next.config.ts` to expose the `output/` directory:

```typescript
// next.config.ts
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/output/:path*',
        destination: '/output/:path*',  // serves from project root/output/
      },
    ];
  },
};
```

Option A is more secure (path traversal protection, file existence check).

## Testing

1. Complete a training job that produces a `.safetensors` file
2. Navigate to results and click a download link
3. Verify the file downloads correctly (not a 404)
