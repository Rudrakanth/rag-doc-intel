import uuid
import json
from src.storage_client import upload_and_get_sas
from src.doc_intel_client import analyze_pdf_from_url
from src.parser.normalize_layout import doc_to_chunks


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
