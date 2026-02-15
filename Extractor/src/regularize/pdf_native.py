"""Native PDF text extraction (PyMuPDF) with font-aware heading detection and text cleanup."""
import re
from typing import List, Tuple

import fitz  # PyMuPDF


def is_text_extractable(path: str, min_chars: int = 50) -> bool:
    """Heuristic to decide if PDF has extractable text (else fallback to OCR)."""
    with fitz.open(path) as doc:
        total = sum(len(page.get_text()) for page in doc)
    return total >= min_chars


# Roman numerals (optional trailing dot) for section headings
_ROMAN_RE = re.compile(r"^\s*[IVXLCDM]+\.?\s", re.IGNORECASE)
# Numbered sections: 1, 1.1, 1.2.3, etc. (require space after)
_NUM_HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+")
# Article / Section style
_ARTICLE_SECTION_RE = re.compile(r"^\s*(Article|Section|Part|Chapter)\s+\d+", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    """Clean PDF text: dehyphenate line breaks, collapse whitespace, strip."""
    if not text:
        return ""
    # Join hyphenated line breaks: "policy-\nrelated" -> "policy-related"
    text = re.sub(r"-\s*\n\s*", "", text)
    # Collapse multiple newlines to double (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _is_heading_by_pattern(line: str) -> bool:
    """True if line looks like a section heading by pattern."""
    line = line.strip()
    if not line or len(line) > 120:
        return False
    first_line = line.split("\n")[0].strip()
    if not first_line:
        return False
    # Numbered: 1, 1.1, 1.2.3
    if _NUM_HEADING_RE.match(first_line):
        return True
    # Article 1, Section 2, Part II
    if _ARTICLE_SECTION_RE.match(first_line):
        return True
    # Roman numeral start
    if _ROMAN_RE.match(first_line):
        return True
    # Short and ends with colon (e.g. "Recommendations:")
    if len(first_line) <= 80 and first_line.endswith(":"):
        return True
    # ALL CAPS and short (common in legal/policy docs)
    if len(first_line) <= 80 and first_line.isupper() and len(first_line) > 2:
        return True
    return False


def _heading_level(text: str) -> int:
    """Infer section level (1–6) from heading text."""
    line = text.strip().split("\n")[0]
    m = re.match(r"^(\d+(\.\d+)*)", line)
    if m:
        depth = len(m.group(1).split("."))
        return min(depth, 6)
    if _ARTICLE_SECTION_RE.match(line) or _ROMAN_RE.match(line):
        return 1
    if line.isupper():
        return 1
    return 2


def _maybe_table(text: str) -> List[List[str]] | None:
    """Heuristic table detection: split lines by pipes/tabs/2+ spaces; require >1 row with 2+ cols."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    rows: List[List[str]] = []
    for ln in lines:
        if "|" in ln:
            cells = [c.strip() for c in ln.split("|") if c.strip()]
        elif "\t" in ln:
            cells = [c.strip() for c in ln.split("\t") if c.strip()]
        else:
            cells = re.split(r"\s{2,}", ln)
            cells = [c.strip() for c in cells if c.strip()]
        if len(cells) < 2:
            return None
        rows.append(cells)
    return rows if len(rows) >= 2 else None


def _get_page_font_stats(page: fitz.Page) -> Tuple[float, float]:
    """Return (median font size, max font size) for text blocks on page."""
    try:
        d = page.get_text("dict")
    except Exception:
        return 12.0, 14.0
    sizes: List[float] = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                s = span.get("size")
                if s is not None and s > 0:
                    sizes.append(float(s))
    if not sizes:
        return 12.0, 14.0
    sizes.sort()
    mid = len(sizes) // 2
    median = sizes[mid] if len(sizes) % 2 else (sizes[mid - 1] + sizes[mid]) / 2
    return median, max(sizes)


def _extract_blocks_with_font(page: fitz.Page, page_num: int) -> List[Tuple[str, float, int]]:
    """Extract (text, font_size, page_num) for each text block. Uses dict for font size."""
    result: List[Tuple[str, float, int]] = []
    try:
        d = page.get_text("dict")
    except Exception:
        # Fallback to plain blocks
        for b in page.get_text("blocks"):
            if len(b) >= 5 and b[4].strip():
                result.append((b[4].strip(), 12.0, page_num))
        return result
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_text_parts: List[str] = []
        max_size = 0.0
        for line in block.get("lines", []):
            line_parts: List[str] = []
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    line_parts.append(t)
                    s = span.get("size")
                    if s is not None:
                        max_size = max(max_size, float(s))
            if line_parts:
                block_text_parts.append(" ".join(line_parts))
        if block_text_parts:
            text = "\n".join(block_text_parts).strip()
            if text:
                result.append((text, max_size if max_size > 0 else 12.0, page_num))
    return result


def extract_pdf_native(path: str) -> Tuple[List[dict], List[str]]:
    """
    Return (sections, pages_text) for text-based PDFs.
    Uses font size to help detect headings and normalizes text for better extraction.
    """
    sections: List[dict] = []
    pages_text: List[str] = []
    all_blocks: List[Tuple[str, float, int]] = []

    with fitz.open(path) as doc:
        for page in doc:
            page_num = page.number + 1
            blocks_with_font = _extract_blocks_with_font(page, page_num)
            all_blocks.extend(blocks_with_font)
            page_text_parts = [_normalize_text(t) for t, _, _ in blocks_with_font if t]
            pages_text.append("\n\n".join(p for p in page_text_parts if p))

    # Compute median font size across document for heading threshold
    font_sizes = [s for _, s, _ in all_blocks if s > 0]
    if font_sizes:
        font_sizes.sort()
        mid = len(font_sizes) // 2
        median_font = font_sizes[mid] if len(font_sizes) % 2 else (font_sizes[mid - 1] + font_sizes[mid]) / 2
        # Heading if font is at least 1.2x median or pattern matches
        heading_font_threshold = median_font * 1.15
    else:
        heading_font_threshold = 14.0

    current_section: dict | None = None
    for text, font_size, page_num in all_blocks:
        text = _normalize_text(text)
        if not text:
            continue
        first_line = text.split("\n")[0].strip()
        is_heading = _is_heading_by_pattern(text) or (font_size >= heading_font_threshold and len(first_line) <= 100)
        table_rows = _maybe_table(text) if not is_heading else None

        if is_heading:
            if current_section:
                sections.append(current_section)
            current_section = {
                "heading": first_line,
                "level": _heading_level(text),
                "paragraphs": [],
                "tables": [],
                "page": page_num,
            }
        elif table_rows:
            if not current_section:
                current_section = {"heading": None, "level": 1, "paragraphs": [], "tables": [], "page": page_num}
            current_section["tables"].append({"rows": table_rows, "page": page_num})
        else:
            if not current_section:
                current_section = {"heading": None, "level": 1, "paragraphs": [], "tables": [], "page": page_num}
            current_section["paragraphs"].append({"text": text, "page": page_num})

    if current_section:
        sections.append(current_section)

    # If no headings were found, treat whole doc as one section (avoid empty sections)
    if not sections and pages_text:
        full_flat = "\n\n".join(pages_text)
        para_blocks, table_blocks = _split_blocks_with_tables(full_flat)
        paragraphs = [{"text": p, "page": 1} for p in para_blocks if p.strip()]
        tables = [{"rows": t, "page": 1} for t in table_blocks]
        sections = [{"heading": None, "level": 1, "paragraphs": paragraphs, "tables": tables, "page": 1}]

    return sections, pages_text


def _split_blocks_with_tables(text: str) -> Tuple[List[str], List[List[List[str]]]]:
    """Split text into paragraphs and table rows. Used when falling back to single section."""
    paragraphs: List[str] = []
    tables: List[List[List[str]]] = []
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) >= 2:
            rows: List[List[str]] = []
            valid = True
            for ln in lines:
                if "|" in ln:
                    cells = [c.strip() for c in ln.split("|") if c.strip()]
                elif "\t" in ln:
                    cells = [c.strip() for c in ln.split("\t") if c.strip()]
                else:
                    cells = re.split(r"\s{2,}", ln)
                    cells = [c.strip() for c in cells if c.strip()]
                if len(cells) < 2:
                    valid = False
                    break
                rows.append(cells)
            if valid and len(rows) >= 2:
                tables.append(rows)
                continue
        paragraphs.append(block.strip())
    return paragraphs, tables
