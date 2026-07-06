import { NextRequest, NextResponse } from 'next/server';
import { runTablePipeline } from '@/lib/pipeline';

// Force Node.js runtime (sharp + onnxruntime-node require native bindings)
export const runtime = 'nodejs';
// Vercel Hobby max is 60 s; Pro is 300 s. We default to 60.
export const maxDuration = 60;
// Always run dynamically — never statically cached
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const file = form.get('file');
    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: 'No file uploaded. Send multipart/form-data with field "file".' },
        { status: 400 }
      );
    }

    // Hard cap at 4.5 MB to stay under Vercel's serverless body limit
    if (file.size > 4.5 * 1024 * 1024) {
      return NextResponse.json(
        { error: 'Image exceeds 4.5 MB. Please downscale before uploading.' },
        { status: 413 }
      );
    }

    const bytes = Buffer.from(await file.arrayBuffer());
    const result = await runTablePipeline(bytes);
    return NextResponse.json(result);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : 'Internal error';
    console.error('[extract-table]', e);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    ok: true,
    endpoint: 'POST /api/extract-table',
    usage:
      'multipart/form-data with a "file" field containing a PNG/JPEG/WebP image (max 4.5 MB)',
  });
}
