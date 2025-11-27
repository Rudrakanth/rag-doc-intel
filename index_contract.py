import os
import glob
from src.pipelines.ingest_pipeline import ingest_contract
from src.indexer.search_indexer import index_chunks, delete_all_documents

INPUT_DIR = "inputdocs"
PROCESSED_DIR = "processed"


def ensure_inputdocs_folder():
    """Create inputdocs/ folder if it doesn't exist."""
    if not os.path.exists(INPUT_DIR):
        print("📁 Creating inputdocs/ folder...")
        os.makedirs(INPUT_DIR)
        print("✔ inputdocs/ folder created. Add PDF files and rerun.")
    else:
        print("📁 inputdocs/ folder already exists.")


def clean_processed_folder():
    """Delete processed JSONL files."""
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        return

    for f in glob.glob(os.path.join(PROCESSED_DIR, "*.jsonl")):
        print(f"🗑 Removing: {f}")
        os.remove(f)


def get_input_files():
    """Return list of PDFs in inputdocs."""
    pdfs = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    if not pdfs:
        print("⚠ No PDF files inside inputdocs/. Add files and rerun.")
    return pdfs


def index_all_documents():

    print("\n=========================")
    print("   RAG INDEXING SYSTEM")
    print("=========================\n")

    ensure_inputdocs_folder()
    clean_processed_folder()

    print("\n🧹 Flushing Azure Search index...")
    delete_all_documents()

    pdf_files = get_input_files()
    if not pdf_files:
        return

    for pdf in pdf_files:
        print(f"\n📄 Processing: {pdf}")
        result = ingest_contract(pdf, model="prebuilt-read")

        doc_id = result["doc_id"]
        chunks = result["chunks"]

        print(f"✔ Ingested doc id: {doc_id}")
        print(f"✔ Total chunks: {len(chunks)}")

        print("📤 Uploading chunks to Azure Search…")
        index_chunks(chunks)

        print(f"✔ Finished indexing {pdf}")

    print("\n🎉 All documents indexed successfully!")


if __name__ == "__main__":
    index_all_documents()
