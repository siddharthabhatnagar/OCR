import { pipeline, env, RawImage } from '@huggingface/transformers';
import Tesseract from 'tesseract.js';
import { mkdirSync } from 'fs';
import { join } from 'path';

// --- transformers.js configuration ---------------------------------------
// Always pull from HF Hub (no local models bundled).
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = false;

// On Vercel, /tmp is the only writable directory and it persists for the
// lifetime of the warm serverless instance. We cache models + worker files
// there so cold starts only happen once per warm cycle.
if (process.env.VERCEL) {
  const tmpCache = '/tmp/hf-cache';
  try { mkdirSync(tmpCache, { recursive: true }); } catch {}
  env.cacheDir = tmpCache;
}

// INT8 (uint8) quantized ONNX — ~30 MB per checkpoint instead of ~110 MB
const MODEL_OPTS = {
  device: 'cpu' as const,
  dtype: 'q8' as const,
};

// Default model IDs — community ONNX exports of Microsoft's TATR family.
// Override via env vars if you want to use a different export.
const DETECTION_MODEL_ID =
  process.env.DETECTION_MODEL_ID || 'Xenova/table-transformer-detection';
const STRUCTURE_MODEL_ID =
  process.env.STRUCTURE_MODEL_ID ||
  'Xenova/table-transformer-structure-recognition-v1.1-all';

// --- lazy singletons ------------------------------------------------------
let detectorPromise: Promise<any> | null = null;
let structurePromise: Promise<any> | null = null;
let workerPromise: Promise<Tesseract.Worker> | null = null;

export function getDetector() {
  if (!detectorPromise) {
    console.log('[models] loading detector:', DETECTION_MODEL_ID);
    detectorPromise = pipeline('object-detection', DETECTION_MODEL_ID, MODEL_OPTS);
  }
  return detectorPromise;
}

export function getStructureRecognizer() {
  if (!structurePromise) {
    console.log('[models] loading structure:', STRUCTURE_MODEL_ID);
    structurePromise = pipeline(
      'object-detection',
      STRUCTURE_MODEL_ID,
      MODEL_OPTS
    );
  }
  return structurePromise;
}

// --- box normalization ----------------------------------------------------
// transformers.js v3 object-detection pipeline returns objects in one of two
// shapes: { boxes, labels, scores } arrays OR a flat list of detections.
// We normalize to a single flat shape.
export interface Box {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  label: string;
  score: number;
}

function normalizeOutput(out: any): Box[] {
  if (!out) return [];

  // shape 1: array containing a single object with arrays
  if (Array.isArray(out) && out.length > 0 && out[0]?.boxes) {
    const o = out[0];
    return o.boxes.map((b: number[], i: number) => ({
      xmin: b[0],
      ymin: b[1],
      xmax: b[2],
      ymax: b[3],
      label: String(o.labels[i] ?? ''),
      score: Number(o.scores[i] ?? 0),
    }));
  }

  // shape 2: object with arrays directly
  if (out.boxes && Array.isArray(out.boxes)) {
    return out.boxes.map((b: number[], i: number) => ({
      xmin: b[0],
      ymin: b[1],
      xmax: b[2],
      ymax: b[3],
      label: String(out.labels[i] ?? ''),
      score: Number(out.scores[i] ?? 0),
    }));
  }

  // shape 3: already flat array of detections
  if (Array.isArray(out)) {
    return out.map((o: any) => ({
      xmin: o.xmin ?? o.bbox?.[0] ?? 0,
      ymin: o.ymin ?? o.bbox?.[1] ?? 0,
      xmax: o.xmax ?? o.bbox?.[2] ?? 0,
      ymax: o.ymax ?? o.bbox?.[3] ?? 0,
      label: String(o.label ?? ''),
      score: Number(o.score ?? 0),
    }));
  }

  return [];
}

export async function detectTables(
  raw: Buffer,
  dims: { W: number; H: number }
): Promise<Box[]> {
  const model = await getDetector();
  const image = new RawImage(raw, dims.W, dims.H, 3);
  const out = await model(image);
  return normalizeOutput(out);
}

export async function recognizeStructure(
  raw: Buffer,
  dims: { W: number; H: number }
): Promise<Box[]> {
  const model = await getStructureRecognizer();
  const image = new RawImage(raw, dims.W, dims.H, 3);
  const out = await model(image);
  return normalizeOutput(out);
}

// --- OCR via tesseract.js -------------------------------------------------
async function getWorker(): Promise<Tesseract.Worker> {
  if (!workerPromise) {
    const lang = process.env.TESSERACT_LANG || 'eng';
    const opts: any = {
      cacheMethod: 'write',
      logger: () => {}, // silence progress spam
    };
    if (process.env.VERCEL) {
      try {
        mkdirSync('/tmp/tess-cache', { recursive: true });
      } catch {}
      opts.cachePath = '/tmp/tess-cache';
      opts.workerPath = join('/tmp/tess-cache', 'worker.min.js');
      opts.corePath = join('/tmp/tess-cache', 'tesseract-core.wasm.js');
      opts.langPath = '/tmp/tess-cache';
    }
    console.log('[models] creating tesseract worker, lang=', lang);
    const worker = await Tesseract.createWorker(lang, 1, opts);
    workerPromise = Promise.resolve(worker);
  }
  return workerPromise;
}

export async function runOCR(pngBuffer: Buffer): Promise<string> {
  const worker = await getWorker();
  const { data } = await worker.recognize(pngBuffer);
  return (data.text || '').replace(/\s+/g, ' ').trim();
}
