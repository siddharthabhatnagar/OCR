"""
Vercel Python serverless function entry point.
Handles GET / and POST / — basic service info.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


def handler(req: BaseHTTPRequestHandler):
    """Vercel calls this function for every request to /."""
    if req.method == "GET":
        body = json.dumps({
            "service": "docling-ocr-api",
            "version": "0.1.0",
            "model": "docling",
            "docs": "/docs",
            "endpoints": ["/health", "/ocr"],
            "ui": "POST a file to /ocr with multipart/form-data",
        }, indent=2).encode()
        req.send_response(200)
        req.send_header("Content-Type", "application/json")
        req.send_header("Content-Length", str(len(body)))
        req.end_headers()
        req.wfile.write(body)
        return

    req.send_response(405)
    req.send_header("Content-Type", "application/json")
    req.end_headers()
    req.wfile.write(b'{"detail":"Method not allowed. GET for info, POST to /ocr for OCR."}')
