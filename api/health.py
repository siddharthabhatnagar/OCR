"""
GET /health — service health check.
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler

from _serverless import cors_headers, send_json
from service.ocr import is_loaded, get_load_error


_START = time.time()


def handler(req: BaseHTTPRequestHandler):
    body = {
        "status": "error" if get_load_error() else ("ok" if is_loaded() else "loading"),
        "model_loaded": is_loaded(),
        "device": "cpu",
        "uptime_s": int(time.time() - _START),
    }
    if get_load_error():
        body["error"] = get_load_error()
    send_json(req, 200, body)
