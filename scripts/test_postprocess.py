"""Sanity test for service.postprocess. Run: python3 scripts/test_postprocess.py"""
import sys, types
sys.modules['loguru'] = types.ModuleType('loguru')
sys.modules['loguru'].logger = type('L', (), {
    'info': print, 'warning': print, 'exception': print
})()

from service.postprocess import split_pages, extract_elements
from service.schemas import PageMeta

# Test 1: page splitting
md = "Page 1 content\n\n---\n\nPage 2 content\n\n---\n\nPage 3 content"
parts = split_pages(md.replace("---", "<PAGE>"), 3)
assert len(parts) == 3, parts
print("[PASS] split_pages")

# Test 2: element extraction
sample = """# Document Title

Some intro paragraph here.

## Section A

Body text under section A.

| Col1 | Col2 |
|------|------|
| 1    | 2    |
| 3    | 4    |

Block equation:
$$E = mc^2$$

Figure: Diagram showing the OCR pipeline.

More body text after the figure.
"""
meta = PageMeta(page_index=0, width_px=1000, height_px=1500, dpi=144)
elements = extract_elements(sample, 0, meta)
types_found = [e.type for e in elements]
print("Element types:", types_found)

assert "heading" in types_found
assert "table" in types_found
assert "equation" in types_found
assert "figure" in types_found
assert "text" in types_found

# Verify specific content
eq = next(e for e in elements if e.type == "equation")
assert "E = mc^2" in eq.content
print("[PASS] extract_elements: all 5 types detected, content correct")

print()
print("ALL TESTS PASSED")
