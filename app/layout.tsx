import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Table OCR — Vercel',
  description:
    'Extract tables from images using Microsoft Table Transformer (INT8 ONNX) + Tesseract.js. Runs inside Vercel serverless, under 250 MB.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
