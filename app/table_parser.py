"""Parse PaddleOCR's table HTML into structured cells with row/col mapping."""
from typing import Dict, Any, List
from html import unescape
from bs4 import BeautifulSoup


def parse_html_table(html: str) -> Dict[str, Any]:
    """
    Convert PaddleOCR's HTML table output into structured cells.

    Handles colspan and rowspan correctly by tracking occupied (row, col)
    positions. Each cell gets:
        { text, row, col, rowspan, colspan }
    """
    if not html:
        return {"rows": 0, "cols": 0, "cells": [], "html": ""}

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table")
    if not table:
        return {"rows": 0, "cols": 0, "cells": [], "html": html}

    cells: List[Dict[str, Any]] = []
    occupied: set = set()
    max_row = -1
    max_col = -1

    for r, tr in enumerate(table.find_all("tr")):
        col = 0
        for cell in tr.find_all(["td", "th"]):
            # Skip positions occupied by a previous rowspan/colspan
            while (r, col) in occupied:
                col += 1

            text = unescape(cell.get_text(separator=" ", strip=True))

            try:
                colspan = max(1, int(cell.get("colspan", 1) or 1))
            except (ValueError, TypeError):
                colspan = 1
            try:
                rowspan = max(1, int(cell.get("rowspan", 1) or 1))
            except (ValueError, TypeError):
                rowspan = 1

            cells.append({
                "text": text,
                "row": r,
                "col": col,
                "rowspan": rowspan,
                "colspan": colspan,
            })

            # Mark all positions this cell occupies
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied.add((r + dr, col + dc))

            max_row = max(max_row, r)
            max_col = max(max_col, col + colspan - 1)
            col += colspan

    return {
        "rows": max_row + 1 if max_row >= 0 else 0,
        "cols": max_col + 1 if max_col >= 0 else 0,
        "cells": cells,
        "html": html,
    }
