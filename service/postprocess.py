"""
Post-process the model's raw markdown output.

The model emits `<PAGE>` markers between pages and pipe-style markdown tables,
LaTeX equations in $...$ / $$...$$, and figure captions starting with
"Figure:". This module:
  1. Splits the raw output on <PAGE> markers into per-page markdown.
  2. Walks each page's markdown and produces typed Element records
     (text / heading / table / equation / figure) for the RAG chunker.
"""
from __future__ import annotations

import re

from .schemas import Element, PageMeta


PAGE_MARKER = "<PAGE>"

_TABLE_RE = re.compile(
    r"((?:^\|[^\n]+\|\s*\n)(?:^\|[\s:|-]+\|\s*\n)?(?:^\|[^\n]+\|\s*\n)+)",
    re.MULTILINE,
)
_BLOCK_EQUATION_RE = re.compile(r"\$\$([^$]+)\$\$", re.DOTALL)
_FIGURE_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:Figure|Fig\.?|Diagram|Chart|Image|Plate)\s*[:\.\-]\s*([^\n]+)",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


# --------------------------------------------------------------------- #
def split_pages(raw_output: str, n_pages: int) -> list[str]:
    """
    Split the model output on <PAGE> markers and return n_pages chunks.

    The model emits: `<PAGE>page0_md<PAGE>page1_md<PAGE>page2_md...`
    """
    parts = raw_output.split(PAGE_MARKER)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == n_pages:
        return parts
    if len(parts) > n_pages:
        return parts[: n_pages - 1] + ["\n\n".join(parts[n_pages - 1:])]
    return parts + [""] * (n_pages - len(parts))


# --------------------------------------------------------------------- #
def extract_elements(
    page_md: str,
    page_index: int,
    page_meta: PageMeta,
) -> list[Element]:
    """
    Walk `page_md` and return typed Element records.

    Order of operations:
      1. Match block equations (greedily, may span lines)
      2. Match tables (multi-line pipe tables)
      3. Match headings and figure captions line-by-line
      4. Remaining lines become 'text' blocks
    """
    elements: list[Element] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= cs or s >= ce) for cs, ce in consumed)

    # --- Block equations ------------------------------------------------
    for m in _BLOCK_EQUATION_RE.finditer(page_md):
        if overlaps(m.start(), m.end()):
            continue
        elements.append(
            Element(
                type="equation",
                content=f"$${m.group(1).strip()}$$",
                page_index=page_index,
            )
        )
        consumed.append((m.start(), m.end()))

    # --- Tables ---------------------------------------------------------
    for m in _TABLE_RE.finditer(page_md):
        if overlaps(m.start(), m.end()):
            continue
        elements.append(
            Element(
                type="table",
                content=m.group(0).strip(),
                page_index=page_index,
            )
        )
        consumed.append((m.start(), m.end()))

    # --- Headings + figure captions + text ------------------------------
    lines = page_md.splitlines()
    cursor = 0
    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            text = "\n".join(text_buf).strip()
            if text:
                elements.append(
                    Element(
                        type="text",
                        content=text,
                        page_index=page_index,
                    )
                )
            text_buf.clear()

    for line in lines:
        start = cursor
        end = cursor + len(line) + 1
        if overlaps(start, end):
            cursor = end
            continue

        h = _HEADING_RE.match(line)
        fig = _FIGURE_RE.match(line)

        if h:
            flush_text()
            elements.append(
                Element(
                    type="heading",
                    content=h.group(2).strip(),
                    page_index=page_index,
                    extra={"level": len(h.group(1))},
                )
            )
            consumed.append((start, end))
        elif fig:
            flush_text()
            elements.append(
                Element(
                    type="figure",
                    content=fig.group(1).strip(),
                    page_index=page_index,
                )
            )
            consumed.append((start, end))
        else:
            text_buf.append(line)
            consumed.append((start, end))
        cursor = end

    flush_text()
    return elements
