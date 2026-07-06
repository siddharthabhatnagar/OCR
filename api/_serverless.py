"""
Shared helpers for all Vercel Python functions.
Vercel injects this via sys.path automatically (functions live in api/).
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


def cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }


def send_json(req: BaseHTTPRequestHandler, status: int, body: dict):
    """Send a JSON response with CORS headers."""
    payload = json.dumps(body, default=str).encode()
    req.send_response(status)
    req.send_header("Content-Type", "application/json")
    for k, v in cors_headers().items():
        req.send_header(k, v)
    req.send_header("Content-Length", str(len(payload)))
    req.end_headers()
    req.wfile.write(payload)


def send_options(req: BaseHTTPRequestHandler):
    """Handle CORS preflight."""
    req.send_response(204)
    for k, v in cors_headers().items():
        req.send_header(k, v)
    req.end_headers()
