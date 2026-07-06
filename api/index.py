"""
Single-entry Vercel Python function.

Vercel's Python runtime expects a top-level `app` variable that is an
ASGI callable. We expose one Starlette ASGI app that routes to all
endpoints. This is cleaner than one file per route.

Routes:
  GET  /            — service info
  GET  /health      — health check
  POST /ocr         — OCR a PDF or image (multipart upload)
  GET  /docs        — Swagger UI
  GET  /openapi.json — OpenAPI spec
"""
from __future__ import annotations

import json
import time
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from service.ocr import get_converter, is_loaded, ocr_bytes


_START = time.time()


# --------------------------------------------------------------------- #
# Route handlers
# --------------------------------------------------------------------- #
async def index(request: Request) -> Response:
    return JSONResponse({
        "service": "docling-ocr-api",
        "version": "0.1.0",
        "model": "docling",
        "docs": "/docs",
        "endpoints": ["/health", "/ocr"],
        "usage": "POST a file to /ocr with multipart/form-data",
    })


async def health(request: Request) -> Response:
    from service.ocr import is_loaded, get_load_error
    body: dict[str, Any] = {
        "status": "error" if get_load_error() else ("ok" if is_loaded() else "loading"),
        "model_loaded": is_loaded(),
        "device": "cpu",
        "uptime_s": int(time.time() - _START),
    }
    if get_load_error():
        body["error"] = get_load_error()
    return JSONResponse(body)


async def ocr(request: Request) -> Response:
    if request.method != "POST":
        return JSONResponse(
            {"detail": "Method not allowed. Use POST with multipart/form-data."},
            status_code=405,
        )

    # Parse multipart form
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse(
            {"detail": f"Malformed multipart body: {e}"},
            status_code=400,
        )

    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse(
            {"detail": "Missing 'file' field in multipart upload"},
            status_code=400,
        )

    # Read file bytes
    file_bytes = await upload.read()
    if not file_bytes:
        return JSONResponse({"detail": "Empty file upload"}, status_code=400)

    filename = upload.filename or "upload.pdf"

    # Run OCR (lazy-loads Docling on first call, ~5 sec)
    try:
        result = ocr_bytes(file_bytes, filename=filename)
    except Exception as e:
        return JSONResponse(
            {"detail": f"OCR failed: {e}"},
            status_code=500,
        )

    return JSONResponse(result.model_dump())


SWAGGER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Docling OCR API — Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        presets: [SwaggerUIBundle.presets.apis],
        layout: 'BaseLayout',
      });
    };
  </script>
</body>
</html>
"""


async def docs(request: Request) -> Response:
    return Response(
        SWAGGER_HTML,
        media_type="text/html",
    )


OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "Docling OCR API",
        "version": "0.1.0",
        "description": (
            "Free OCR API for RAG ingestion, powered by IBM Docling. "
            "Pure CPU, runs on Vercel free tier. No GPU quota."
        ),
    },
    "paths": {
        "/health": {
            "get": {
                "summary": "Health check",
                "responses": {
                    "200": {
                        "description": "Service health",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthResponse"},
                            }
                        },
                    }
                },
            }
        },
        "/ocr": {
            "post": {
                "summary": "OCR a PDF or image",
                "description": (
                    "Upload a PDF or image (PNG/JPEG/WebP/BMP) and get back "
                    "markdown plus typed elements (text/heading/table/equation/figure) "
                    "per page, suitable for RAG chunking."
                ),
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string", "format": "binary"},
                                },
                                "required": ["file"],
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "OCR result",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/OcrResponse"},
                            }
                        },
                    },
                    "400": {"description": "Bad request (empty or malformed upload)"},
                    "500": {"description": "OCR processing error"},
                },
            }
        },
    },
    "components": {
        "schemas": {
            "HealthResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok", "loading", "error"]},
                    "model_loaded": {"type": "boolean"},
                    "device": {"type": "string"},
                    "uptime_s": {"type": "integer"},
                },
            },
            "OcrResponse": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string"},
                    "pages": {"type": "array", "items": {"$ref": "#/components/schemas/PageResult"}},
                    "total_pages": {"type": "integer"},
                    "elapsed_ms": {"type": "integer"},
                    "model": {"type": "string"},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            },
            "PageResult": {
                "type": "object",
                "properties": {
                    "page": {"$ref": "#/components/schemas/PageMeta"},
                    "markdown": {"type": "string"},
                    "elements": {"type": "array", "items": {"$ref": "#/components/schemas/Element"}},
                },
            },
            "PageMeta": {
                "type": "object",
                "properties": {
                    "page_index": {"type": "integer"},
                    "width_px": {"type": "integer"},
                    "height_px": {"type": "integer"},
                    "dpi": {"type": "integer"},
                },
            },
            "Element": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["text", "table", "equation", "figure", "heading", "list"],
                    },
                    "content": {"type": "string"},
                    "page_index": {"type": "integer"},
                },
            },
        }
    },
}


async def openapi(request: Request) -> Response:
    return JSONResponse(OPENAPI_SPEC)


# --------------------------------------------------------------------- #
# Build the Starlette ASGI app
# --------------------------------------------------------------------- #
# Vercel's Python runtime looks for a top-level `app` variable that is an
# ASGI callable. Starlette apps are ASGI callables, so this works.

routes = [
    Route("/", index, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route("/ocr", ocr, methods=["POST", "OPTIONS"]),
    Route("/docs", docs, methods=["GET"]),
    Route("/openapi.json", openapi, methods=["GET"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    ),
]

app = Starlette(routes=routes, middleware=middleware)
