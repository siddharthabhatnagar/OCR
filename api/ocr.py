"""
POST /ocr — OCR a PDF or image upload (multipart/form-data).

Vercel Python functions receive the raw request. We parse multipart
manually because Python's cgi module is removed in 3.13+.
"""
from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler

from _serverless import send_json
from service.ocr import ocr_bytes


def handler(req: BaseHTTPRequestHandler):
    if req.method != "POST":
        send_json(req, 405, {"detail": "Method not allowed. Use POST with multipart/form-data."})
        return

    ctype = req.headers.get("Content-Type", "")
    if "multipart/form-data" not in ctype:
        send_json(req, 415, {
            "detail": "Expected multipart/form-data with a 'file' field"
        })
        return

    # Read the full body
    length = int(req.headers.get("Content-Length", 0))
    body = req.rfile.read(length) if length else b""

    # Parse multipart
    try:
        filename, file_bytes = parse_multipart(body, ctype)
    except ValueError as e:
        send_json(req, 400, {"detail": f"Malformed multipart body: {e}"})
        return

    if not file_bytes:
        send_json(req, 400, {"detail": "Empty file upload"})
        return

    # Run OCR
    try:
        result = ocr_bytes(file_bytes, filename=filename or "upload.pdf")
    except Exception as e:
        send_json(req, 500, {"detail": f"OCR failed: {e}"})
        return

    send_json(req, 200, result.model_dump())


# --------------------------------------------------------------------- #
# Multipart parser
# --------------------------------------------------------------------- #
def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes]:
    """
    Minimal multipart/form-data parser. Extracts the first 'file' field.
    Returns (filename, file_bytes).
    """
    # Get boundary from Content-Type: multipart/form-data; boundary=xxx
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip('"')
            break
    if not boundary:
        raise ValueError("No boundary in Content-Type")

    boundary_bytes = ("--" + boundary).encode()

    # Split on boundary
    chunks = body.split(boundary_bytes)
    for chunk in chunks:
        # Skip preamble/epilogue
        if not chunk or chunk in (b"--", b"--\r\n", b"\r\n"):
            continue

        # Strip leading CRLF
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]

        # Split headers and content
        if b"\r\n\r\n" not in chunk:
            continue
        header_block, content = chunk.split(b"\r\n\r\n", 1)

        # Strip trailing CRLF
        if content.endswith(b"\r\n"):
            content = content[:-2]

        # Parse headers
        headers = {}
        for line in header_block.split(b"\r\n"):
            if b":" in line:
                k, v = line.split(b":", 1)
                headers[k.strip().lower().decode()] = v.strip().decode()

        # Look for 'name="file"' and extract filename
        disp = headers.get("content-disposition", "")
        if 'name="file"' not in disp:
            continue

        filename = ""
        if 'filename="' in disp:
            start = disp.index('filename="') + len('filename="')
            end = disp.index('"', start)
            filename = disp[start:end]

        return filename, content

    raise ValueError("No 'file' field in multipart body")
