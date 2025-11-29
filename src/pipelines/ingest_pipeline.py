import json
import uuid
from typing import BinaryIO, Dict, List, Optional

from src.doc_intel_client import analyze_pdf_from_url
from src.parser.normalize_layout import doc_to_chunks
from src.storage_client import (
    generate_sas_url,
    upload_and_get_sas,
    upload_pdf_fileobj,
)


def ingest_contract(local_pdf_path: str, model="prebuilt-layout"):
    """
    Full ingestion pipeline for commercial lease contracts:
    1. Upload PDF to Azure Blob
    2. Generate SAS URL
    3. Analyze using Document Intelligence (prebuilt-contract)
    4. Convert DI result -> structured chunks
    5. Save chunks into processed/<doc_id>.jsonl

    Returns:
        {
            "doc_id": <uuid>,
            "chunks": <list of chunk dicts>
        }
    """

    # -----------------------------
    # 1. Upload PDF → Get SAS URL
    # -----------------------------
    print("Uploading PDF…")
    sas_url = upload_and_get_sas(local_pdf_path)

    # Generate document id
    doc_id = str(uuid.uuid4())
    source_file = local_pdf_path.split("/")[-1]

    # -----------------------------
    # 2. Analyze using prebuilt-contract
    # -----------------------------
    print("Analyzing contract using model… " + model)
    result = analyze_pdf_from_url(sas_url, model)

    # -----------------------------
    # 3. Convert into semantic chunks
    # -----------------------------
    print("Creating chunks…")
    chunks = doc_to_chunks(result, doc_id, source_file)

    # -----------------------------
    # 4. Save chunks locally
    # -----------------------------
    output_path = f"processed/{doc_id}.jsonl"

    with open(output_path, "w") as f:
        for ch in chunks:
            f.write(json.dumps(ch) + "\n")

    print(f"Saved processed chunks → {output_path}")

    return {
        "doc_id": doc_id,
        "chunks": chunks
    }


def ingest_contract_filelike(
    file_obj: BinaryIO,
    filename: str,
    doc_id: Optional[str] = None,
    model: str = "prebuilt-layout",
) -> Dict[str, List[Dict]]:
    """
    Variant of ingest_contract for in-memory uploads (Streamlit, FastAPI).
    Uploads bytes to blob storage, runs DI, and returns chunks.
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    file_obj.seek(0)
    blob_name = upload_pdf_fileobj(file_obj, filename, doc_id=doc_id)
    sas_url = generate_sas_url(blob_name)

    result = analyze_pdf_from_url(sas_url, model)
    chunks = doc_to_chunks(result, doc_id, filename)

    return {"doc_id": doc_id, "chunks": chunks, "blob_name": blob_name}
