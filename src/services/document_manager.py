import uuid
from typing import BinaryIO, Dict, List, Optional

from src.doc_intel_client import analyze_pdf_from_url
from src.indexer.search_indexer import (
    count_chunks_for_doc,
    delete_documents_by_doc_id,
    index_chunks,
)
from src.parser.normalize_layout import doc_to_chunks
from src.pipelines.ingest_pipeline import ingest_contract_filelike
from src.storage_client import (
    delete_blob,
    generate_sas_url,
    list_blobs,
    set_blob_metadata,
)


def _ensure_doc_id(blob_name: str, metadata: Dict[str, str]) -> str:
    doc_id = metadata.get("doc_id")
    if not doc_id:
        doc_id = str(uuid.uuid4())
        new_meta = metadata.copy()
        new_meta["doc_id"] = doc_id
        new_meta.setdefault("original_name", blob_name)
        set_blob_metadata(blob_name, new_meta)
    return doc_id


def get_document_status() -> List[Dict]:
    """
    Return a list of documents with blob + indexing metadata.
    """
    docs = []
    for blob in list_blobs():
        metadata = blob.get("metadata") or {}
        doc_id = metadata.get("doc_id")
        original_name = metadata.get("original_name") or blob["name"]
        indexed_chunks = count_chunks_for_doc(doc_id) if doc_id else 0
        status = "Indexed" if indexed_chunks else "Uploaded"

        docs.append(
            {
                "doc_id": doc_id,
                "blob_name": blob["name"],
                "filename": original_name,
                "last_modified": blob.get("last_modified"),
                "size": blob.get("size"),
                "indexed_chunks": indexed_chunks,
                "status": status,
            }
        )

    docs.sort(key=lambda d: d.get("last_modified") or "", reverse=True)
    return docs


def upload_and_index(file_obj: BinaryIO, filename: str) -> Dict[str, str]:
    """
    Upload a new PDF to blob storage and index it into Azure Search.
    """
    ingestion = ingest_contract_filelike(file_obj, filename)
    index_chunks(ingestion["chunks"])
    return {"doc_id": ingestion["doc_id"], "blob_name": ingestion["blob_name"]}


def reindex_document(blob_name: str, metadata: Dict[str, str], filename: str) -> int:
    """
    Re-run ingestion on an existing blob and replace index entries.
    """
    doc_id = _ensure_doc_id(blob_name, metadata)
    sas_url = generate_sas_url(blob_name)
    result = analyze_pdf_from_url(sas_url, model="prebuilt-layout")
    chunks = doc_to_chunks(result, doc_id, filename)

    delete_documents_by_doc_id(doc_id)
    index_chunks(chunks)
    return len(chunks)


def remove_document(blob_name: str, doc_id: Optional[str]):
    """
    Delete blob and associated indexed chunks.
    """
    if doc_id:
        delete_documents_by_doc_id(doc_id)
    delete_blob(blob_name)
