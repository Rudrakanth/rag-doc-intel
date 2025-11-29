from datetime import datetime, timedelta
from typing import BinaryIO, Dict, List

from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas

from .config import AZURE_BLOB_CONN_STR, AZURE_BLOB_CONTAINER_RAW


blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
container_client = blob_service.get_container_client(AZURE_BLOB_CONTAINER_RAW)


def upload_pdf(local_file_path: str, blob_name: str = None, metadata: Dict[str, str] = None) -> str:
    """
    Uploads a PDF from Mac local storage to Azure Blob Storage.

    Parameters:
        local_file_path (str): Path to local PDF file.
        blob_name (str): Optional blob name. If omitted, filename is used.

    Returns:
        str: Blob URL (not SAS)
    """
    if blob_name is None:
        blob_name = local_file_path.split("/")[-1]

    blob_client = container_client.get_blob_client(blob_name)

    with open(local_file_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True, metadata=metadata)

    return blob_client.url


def upload_pdf_fileobj(file_obj: BinaryIO, filename: str, doc_id: str) -> str:
    """
    Upload a PDF from an in-memory file-like object and attach metadata.
    """
    blob_client = container_client.get_blob_client(filename)
    metadata = {"doc_id": doc_id, "original_name": filename}
    blob_client.upload_blob(file_obj, overwrite=True, metadata=metadata)
    return blob_client.blob_name


def generate_sas_url(blob_name: str, expiry_hours: int = 2) -> str:
    """
    Generates a SAS URL so Document Intelligence can access the file.

    Parameters:
        blob_name (str): Blob name in raw-pdfs container.
        expiry_hours (int): Validity of SAS token.

    Returns:
        str: Full SAS URL
    """
    blob_client = container_client.get_blob_client(blob_name)

    account_key = _extract_account_key(AZURE_BLOB_CONN_STR)

    sas_token = generate_blob_sas(
        account_name=blob_service.account_name,
        account_key=account_key, 
        container_name=AZURE_BLOB_CONTAINER_RAW,
        blob_name=blob_name,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
    )

    sas_url = f"{blob_client.url}?{sas_token}"
    return sas_url


def upload_and_get_sas(local_file: str) -> str:
    """
    Convenience method:
    1. Upload local PDF
    2. Return SAS URL for Document Intelligence
    """
    blob_url = upload_pdf(local_file)
    blob_name = blob_url.split("/")[-1]
    return generate_sas_url(blob_name)

def _extract_account_key(conn_str: str):
    parts = conn_str.split(";")
    for p in parts:
        if p.startswith("AccountKey="):
            return p.replace("AccountKey=", "")
    raise ValueError("AccountKey not found in connection string.")


def list_blobs() -> List[Dict]:
    """
    Return blob metadata for the raw container.
    """
    blobs = []
    for blob in container_client.list_blobs():
        blobs.append(
            {
                "name": blob.name,
                "metadata": blob.metadata or {},
                "size": blob.size,
                "last_modified": blob.last_modified,
            }
        )
    return blobs


def delete_blob(blob_name: str):
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.delete_blob(delete_snapshots="include")


def set_blob_metadata(blob_name: str, metadata: Dict[str, str]):
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.set_blob_metadata(metadata)
