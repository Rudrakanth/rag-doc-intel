from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from .config import (
    AZURE_FORM_RECOGNIZER_ENDPOINT,
    AZURE_FORM_RECOGNIZER_KEY,
)


# -----------------------------------------------
# Initialize v4 Document Intelligence Client
# -----------------------------------------------
client = DocumentIntelligenceClient(
    endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
    credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
)


# -----------------------------------------------
# Analyze PDF from SAS URL
# -----------------------------------------------
def analyze_pdf_from_url(blob_sas_url: str, model: str = "prebuilt-layout"):
    """
    Analyze a PDF using Azure Document Intelligence v4 via SAS URL.

    Required:
        - Model name (prebuilt-document, prebuilt-layout, custom-model-id)
        - AnalyzeDocumentRequest(url_source=<sas_url>)
    """

    poller = client.begin_analyze_document(
        model,
        AnalyzeDocumentRequest(url_source=blob_sas_url)
    )

    result = poller.result()
    return result


# -----------------------------------------------
# Analyze PDF from local bytes (Mac development)
# -----------------------------------------------
def analyze_pdf_bytes(file_bytes: bytes, model: str = "prebuilt-layout"):
    """
    Analyze a PDF using raw bytes read from a local file.

    Useful for:
        - Testing before uploading to Azure Blob Storage
        - Debugging corruption issues
    """

    poller = client.begin_analyze_document(
        model,
        file_bytes
    )

    result = poller.result()
    return result
