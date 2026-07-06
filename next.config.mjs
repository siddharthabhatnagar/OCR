/**
 * Next.js config for Table OCR — Next.js 14.2.x compatible.
 *
 * Critical fix: in Next.js 14.2 the option is called
 * `serverComponentsExternalPackages` (NOT `serverExternalPackages`, which
 * only exists in Next.js 15+). Using the wrong name silently leaves native
 * packages bundled, which makes webpack try to parse `.node` binary files
 * and the build fails with:
 *
 *   "You may need an appropriate loader to handle this file type, currently
 *    no loaders are configured to process this file."
 *
 * Strategy:
 *   1. `serverComponentsExternalPackages` — primary mechanism: tells Next.js
 *      to leave these as runtime `require()` calls instead of bundling.
 *   2. `webpack.externals` — belt-and-suspenders for any sub-paths that
 *      bypass the package-level externalization.
 *   3. `webpack.module.rules` for `.node` files — final fallback: emit the
 *      binary as a separate asset via asset/resource, never parse as JS.
 */

const NATIVE_PACKAGES = [
  '@huggingface/transformers',
  'onnxruntime-node',
  'onnxruntime-common',
  'onnxruntime-web',
  'tesseract.js',
  'tesseract.js-core',
  'sharp',
  'workerpool',
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Next.js 14.2 name — must use this exact key, NOT `serverExternalPackages`.
  serverComponentsExternalPackages: NATIVE_PACKAGES,

  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },

  images: {
    formats: ['image/avif', 'image/webp'],
  },

  webpack: (config, { isServer }) => {
    // (a) Add externals — both server and client. On the server these become
    //     runtime require() calls. On the client they're stubbed (we never
    //     import them client-side anyway).
    if (isServer) {
      const existing = config.externals || [];
      config.externals = [
        ...existing,
        ...NATIVE_PACKAGES.map((pkg) => `${pkg}`),
      ];
    }

    // (b) Final fallback: if webpack still walks into a `.node` binary,
    //     emit it as a separate asset via asset/resource. This is the
    //     built-in webpack 5 loader — no npm install needed.
    config.module.rules.push({
      test: /\.node$/,
      type: 'asset/resource',
      generator: {
        filename: 'native-binaries/[name].[ext]',
      },
    });

    // (c) Some transformers.js sub-imports reference Node built-ins. Make
    //     sure webpack doesn't try to polyfill them on the server.
    if (!config.resolve.fallback) {
      config.resolve.fallback = {};
    }
    Object.assign(config.resolve.fallback, {
      fs: false,
      path: false,
      crypto: false,
      stream: false,
      worker_threads: false,
      perf_hooks: false,
      util: false,
      os: false,
    });

    return config;
  },
};

export default nextConfig;
