'use client';
import { useCallback, useRef, useState } from 'react';

interface Cell {
  text: string;
  row: number;
  col: number;
}
interface ExtractResult {
  rows: number;
  cols: number;
  cells: Cell[];
  html: string;
  elapsed_ms: number;
}

const MAX_BYTES = 4 * 1024 * 1024; // 4 MB (Vercel serverless body limit is 4.5 MB)

export default function Home() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ExtractResult | null>(null);
  const [drag, setDrag] = useState(false);

  const onSelect = (f: File | null) => {
    if (!f) return;
    if (f.size > MAX_BYTES) {
      setError(`File too large (${(f.size / 1024 / 1024).toFixed(2)} MB). Vercel serverless body limit is 4.5 MB — please use an image under 4 MB.`);
      return;
    }
    if (!/image\/(png|jpe?g|webp)/.test(f.type)) {
      setError('Please upload a PNG, JPEG, or WebP image.');
      return;
    }
    setError('');
    setResult(null);
    setFile(f);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    onSelect(e.dataTransfer.files?.[0] ?? null);
  }, []);

  const onExtract = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/extract-table', {
        method: 'POST',
        body: form,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Extraction failed');
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1>Table OCR</h1>
      <p className="subtitle">
        Microsoft Table Transformer (INT8 ONNX) + Tesseract.js — runs entirely
        inside a Vercel serverless function, under the 250&nbsp;MB limit. No
        external API calls.
      </p>

      <div
        className={`upload-zone${drag ? ' drag' : ''}`}
        onClick={() => fileRef.current?.click()}
        onDrop={onDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(e) => onSelect(e.target.files?.[0] ?? null)}
        />
        <p>
          <strong>Click to upload</strong> or drag &amp; drop
        </p>
        <p>PNG / JPEG / WebP — max 4&nbsp;MB</p>
      </div>

      {preview && (
        <div className="preview">
          <img src={preview} alt="preview" />
          <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
            <button onClick={onExtract} disabled={loading}>
              {loading ? 'Extracting…' : 'Extract Table'}
            </button>
            {file && (
              <span style={{ alignSelf: 'center', color: '#888', fontSize: 13 }}>
                {file.name} — {(file.size / 1024).toFixed(0)} KB
              </span>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div className="status loading">
          Running table detection → structure recognition → cell-by-cell OCR.
          First call after a cold start may take 10–20&nbsp;s while models are
          downloaded to <code>/tmp</code> and the OCR worker initialises.
          Subsequent calls on a warm instance are much faster.
        </div>
      )}
      {error && <div className="status error">⚠ {error}</div>}

      {result && (
        <div className="results">
          <div className="status success">
            ✅ Extracted {result.rows} rows × {result.cols} cols in{' '}
            {(result.elapsed_ms / 1000).toFixed(2)}&nbsp;s
          </div>

          <h3>Rendered table</h3>
          <div dangerouslySetInnerHTML={{ __html: result.html }} />

          <h3>JSON cells</h3>
          <pre>{JSON.stringify(result.cells, null, 2)}</pre>
        </div>
      )}

      <p className="meta">
        Models: <code>Xenova/table-transformer-detection</code> +
        <code> Xenova/table-transformer-structure-recognition-v1.1-all</code>{' '}
        (uint8 quantized ONNX, ~30&nbsp;MB each) and{' '}
        <code>tesseract.js</code> for cell OCR. Total runtime weight
        ~125&nbsp;MB — well under Vercel&apos;s 250&nbsp;MB uncompressed
        serverless limit.
      </p>
    </main>
  );
}
