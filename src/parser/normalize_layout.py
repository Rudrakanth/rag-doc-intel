import os
import re

def extract_full_text(result):
    """
    Best Layout extraction: iterate over pages → lines → content.
    Returns the full text in reading order.
    """
    lines = []

    for page in result.pages:
        for line in page.lines:
            if line.content:
                lines.append(line.content.strip())

    return "\n".join(lines)


def split_into_chunks(text, max_chars=2500):
    """
    Split text intelligently into RAG chunks.
    """
    chunks = []
    current = ""

    for paragraph in text.split("\n"):
        if len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current.strip())
            current = paragraph + "\n"
        else:
            current += paragraph + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


def doc_to_chunks(result, doc_id, source_file, max_chars=2500):
    chunks = []

    # -----------------------------
    # 1. Extract PAGE-LEVEL TEXT
    # -----------------------------
    page_text_map = {}   # page_num -> text
    for page in result.pages:
        text = ""
        if hasattr(page, "lines") and page.lines:
            for ln in page.lines:
                text += ln.content.strip() + "\n"
        page_text_map[page.page_number] = text.strip()

    # -----------------------------
    # 2. Extract STRUCTURE (sections, paragraphs)
    # -----------------------------
    # DI v4 Layout API exposes: result.paragraphs, result.styles, result.tables, etc.

    paragraphs = []
    if hasattr(result, "paragraphs") and result.paragraphs:
        for p in result.paragraphs:
            pg = p.bounding_regions[0].page_number if p.bounding_regions else None
            paragraphs.append({
                "role": p.role or "body",
                "text": p.content,
                "page": pg
            })

    # -----------------------------
    # 3. Extract TABLES (turn each row into readable text)
    # -----------------------------
    table_snippets = []
    if hasattr(result, "tables") and result.tables:
        for t in result.tables:
            rows = {}
            for cell in t.cells:
                rows.setdefault(cell.row_index, {})
                rows[cell.row_index][cell.column_index] = cell.content

            table_text = "TABLE:\n"
            for r in sorted(rows.keys()):
                row_text = " | ".join([rows[r].get(c, "") for c in sorted(rows[r].keys())])
                table_text += row_text + "\n"

            pg = t.bounding_regions[0].page_number if t.bounding_regions else None

            table_snippets.append({
                "page": pg,
                "text": table_text.strip()
            })

    # -----------------------------
    # 4. Extract KEY-VALUE PAIRS (checkboxes, key/value fields)
    # -----------------------------
    kv_pairs = []
    if hasattr(result, "key_value_pairs") and result.key_value_pairs:
        for kv in result.key_value_pairs:
            key = kv.key.content if kv.key else None
            val = kv.value.content if kv.value else None
            pg = kv.key.bounding_regions[0].page_number if kv.key and kv.key.bounding_regions else None

            kv_pairs.append({
                "page": pg,
                "text": f"{key}: {val}"
            })

    # -----------------------------
    # 5. BUILD THE RAW TEXT PER PAGE
    # -----------------------------
    page_blocks = []

    for page_num, text in page_text_map.items():
        block = f"PAGE {page_num}\n{text}"
        
        # Add structured paragraphs
        para_text = "\n".join(
            [p["text"] for p in paragraphs if p["page"] == page_num]
        )
        if para_text.strip():
            block += "\n\nPARAGRAPHS:\n" + para_text

        # Add tables
        tbl_text = "\n".join(
            [t["text"] for t in table_snippets if t["page"] == page_num]
        )
        if tbl_text.strip():
            block += "\n\n" + tbl_text

        # Add KV pairs
        kv_text = "\n".join(
            [k["text"] for k in kv_pairs if k["page"] == page_num]
        )
        if kv_text.strip():
            block += "\n\nKEY-VALUE PAIRS:\n" + kv_text

        page_blocks.append({
            "page": page_num,
            "text": block.strip()
        })

    # -----------------------------
    # 6. Chunk Each Page
    # -----------------------------
    def split_text(text, max_chars=2000):
        text = re.sub(r"\n{3,}", "\n\n", text)
        parts = []
        while len(text) > max_chars:
            split_at = text.rfind("\n", 0, max_chars)
            split_at = split_at if split_at != -1 else max_chars
            parts.append(text[:split_at])
            text = text[split_at:]
        parts.append(text)
        return parts

    # -----------------------------
    # 7. BUILD FINAL RAG CHUNKS
    # -----------------------------
    for block in page_blocks:
        sub_chunks = split_text(block["text"], max_chars=max_chars)
        for idx, sub in enumerate(sub_chunks):
            chunks.append({
                "id": f"{doc_id}-p{block['page']}-c{idx}",
                "doc_id": doc_id,
                "source_file": source_file,
                "page_number": block["page"],
                "content": sub,
                "section_title": f"Page {block['page']}",
            })

    return chunks
