/** @type {import('next').NextConfig} */
const nextConfig = {
  serverExternalPackages: [
    '@huggingface/transformers',
    'onnxruntime-node',
    'tesseract.js',
    'sharp',
  ],
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
  // sharp is already provided by Next.js, but we want explicit control for the API route
  images: {
    formats: ['image/avif', 'image/webp'],
  },
};

export default nextConfig;
