import sharp from 'sharp';
import {
  detectTables,
  recognizeStructure,
  runOCR,
  type Box,
} from './models';

export interface Cell {
  text: string;
  row: number;
  col: number;
}
export interface PipelineResult {
  rows: number;
  cols: number;
  cells: Cell[];
  html: string;
  elapsed_ms: number;
  table_bbox?: { xmin: number; ymin: number; xmax: number; ymax: number };
}

const isRow = (b: Box) => b.label.toLowerCase().includes('row');
const isCol = (b: Box) => b.label.toLowerCase().includes('column');
const isHeader = (b: Box) =>
  b.label.toLowerCase().includes('header') && isRow(b);

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

export async function runTablePipeline(
  imageBytes: Buffer
): Promise<PipelineResult> {
  const start = Date.now();

  // --- load image metadata --------------------------------------------------
  const img = sharp(imageBytes, { failOn: 'none' });
  const meta = await img.metadata();
  const W = meta.width || 0;
  const H = meta.height || 0;
  if (!W || !H) throw new Error('Invalid or unreadable image');

  // raw RGB pixels for transformers.js
  const raw = await sharp(imageBytes)
    .removeAlpha()
    .toFormat('raw')
    .toBuffer();

  // --- 1) table detection ---------------------------------------------------
  const detections = await detectTables(raw, { W, H });
  const tables = detections
    .filter((b) => b.label.toLowerCase() === 'table')
    .sort((a, b) => b.score - a.score);
  if (tables.length === 0) {
    // Fall back to the whole image as a single table
    console.warn('[pipeline] no table detected — using whole image');
  }
  const table = tables[0] ?? {
    xmin: 0,
    ymin: 0,
    xmax: W,
    ymax: H,
    label: 'table',
    score: 0,
  };

  // --- 2) crop to the table region -----------------------------------------
  const cropLeft = clamp(Math.floor(table.xmin), 0, W - 1);
  const cropTop = clamp(Math.floor(table.ymin), 0, H - 1);
  const cropW = clamp(
    Math.ceil(table.xmax - table.xmin),
    1,
    W - cropLeft
  );
  const cropH = clamp(
    Math.ceil(table.ymax - table.ymin),
    1,
    H - cropTop
  );

  const cropped = sharp(imageBytes).extract({
    left: cropLeft,
    top: cropTop,
    width: cropW,
    height: cropH,
  });
  const cropMeta = await cropped.metadata();
  const cropW_ = cropMeta.width || cropW;
  const cropH_ = cropMeta.height || cropH;
  const cropRaw = await cropped.removeAlpha().toFormat('raw').toBuffer();

  // --- 3) structure recognition --------------------------------------------
  const structs = await recognizeStructure(cropRaw, {
    W: cropW_,
    H: cropH_,
  });

  const rows = structs
    .filter(isRow)
    .sort((a, b) => a.ymin - b.ymin);
  const cols = structs
    .filter(isCol)
    .sort((a, b) => a.xmin - b.xmin);
  const headerRowCount = structs.filter(isHeader).length;

  if (rows.length === 0 || cols.length === 0) {
    throw new Error(
      `Could not recognize table structure (found ${rows.length} rows, ${cols.length} columns). Try a clearer image.`
    );
  }

  // --- 4) cell-by-cell OCR -------------------------------------------------
  const cells: Cell[] = [];
  for (let r = 0; r < rows.length; r++) {
    for (let c = 0; c < cols.length; c++) {
      const row = rows[r];
      const col = cols[c];
      const xmin = clamp(Math.floor(col.xmin), 0, cropW_ - 1);
      const ymin = clamp(Math.floor(row.ymin), 0, cropH_ - 1);
      const xmax = clamp(Math.ceil(col.xmax), xmin + 1, cropW_);
      const ymax = clamp(Math.ceil(row.ymax), ymin + 1, cropH_);
      const w = Math.max(1, xmax - xmin);
      const h = Math.max(1, ymax - ymin);

      const cellPng = await sharp(cropRaw, {
        raw: { width: cropW_, height: cropH_, channels: 3 },
      })
        .extract({ left: xmin, top: ymin, width: w, height: h })
        .png()
        .toBuffer();

      const text = await runOCR(cellPng);
      cells.push({ text, row: r, col: c });
    }
  }

  // --- 5) build HTML table --------------------------------------------------
  let html = '<table>';
  for (let r = 0; r < rows.length; r++) {
    html += '<tr>';
    for (let c = 0; c < cols.length; c++) {
      const cell = cells.find((x) => x.row === r && x.col === c);
      const content = cell?.text || '';
      const tag = r < headerRowCount ? 'th' : 'td';
      html += `<${tag}>${escapeHtml(content)}</${tag}>`;
    }
    html += '</tr>';
  }
  html += '</table>';

  return {
    rows: rows.length,
    cols: cols.length,
    cells,
    html,
    elapsed_ms: Date.now() - start,
    table_bbox: {
      xmin: table.xmin,
      ymin: table.ymin,
      xmax: table.xmax,
      ymax: table.ymax,
    },
  };
}
