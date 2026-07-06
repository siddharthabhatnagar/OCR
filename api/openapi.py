"""
GET /openapi.json — OpenAPI 3.0 spec for the OCR API.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


SPEC = {
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
                    "415": {"description": "Unsupported file type"},
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
                    "markdown": {"type": "string", "description": "Full document markdown"},
                    "pages": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/PageResult"},
                    },
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
                    "elements": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Element"},
                    },
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
                    "bbox": {"$ref": "#/components/schemas/BoundingBox"},
                    "extra": {"type": "object"},
                },
            },
            "BoundingBox": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "w": {"type": "number"},
                    "h": {"type": "number"},
                },
            },
        }
    },
}


def handler(req: BaseHTTPRequestHandler):
    body = json.dumps(SPEC, indent=2).encode()
    req.send_response(200)
    req.send_header("Content-Type", "application/json")
    req.send_header("Content-Length", str(len(body)))
    req.end_headers()
    req.wfile.write(body)
