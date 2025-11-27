from typing import List, Dict
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
)
from openai import AzureOpenAI

from src.config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_EMBED_DEPLOYMENT,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_EMBED_ENDPOINT,
)
from src.config import openai_client, AZURE_OPENAI_CHAT_DEPLOYMENT

# -----------------------------
# Azure clients
# -----------------------------
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

aoai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_EMBED_ENDPOINT,
)


# -----------------------------
# Create index (if needed)
# -----------------------------
def create_index_if_not_exists(embedding_dim: int = 1536):
    existing = [idx.name for idx in index_client.list_indexes()]
    if AZURE_SEARCH_INDEX_NAME in existing:
        print(f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists.")
        return

    print(f"Creating index '{AZURE_SEARCH_INDEX_NAME}'…")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
        SearchableField(name="section_title", type=SearchFieldDataType.String),
        SearchField(
            name="tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True
        ),

        # Contract metadata
        SimpleField(name="contract_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="tenant_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="owner_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="property_location", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="gla", type=SearchFieldDataType.Double, filterable=True),
        SimpleField(name="lease_amount", type=SearchFieldDataType.Double, filterable=True),
        SimpleField(name="rent_per_sqft", type=SearchFieldDataType.Double, filterable=True),
        SimpleField(name="start_date", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="end_date", type=SearchFieldDataType.String, filterable=True),

        # VECTOR FIELD — FINAL FIX
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name="my-vector-profile",   # <── REQUIRED FIX
        ),
    ]

    # Matching vector search config
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-config",
                kind="hnsw"
            )
        ],
        profiles=[
            {
                "name": "my-vector-profile",       # <── Profile name
                "algorithm": "my-hnsw-config"      # <── Algorithm must match
            }
        ]
    )

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    index_client.create_index(index)
    print(f"Index '{AZURE_SEARCH_INDEX_NAME}' created successfully.")


# -----------------------------
# Embedding helper
# -----------------------------
def embed_text(text: str) -> List[float]:
    resp = aoai_client.embeddings.create(
        model=AZURE_OPENAI_EMBED_DEPLOYMENT,
        input=text,
    )
    return resp.data[0].embedding


# -----------------------------
# Index chunks into Azure Search
# -----------------------------
def index_chunks(chunks: List[Dict]):
    """
    Takes list of chunks (from doc_to_chunks) and uploads to Azure Search.
    """
    docs = []

    for ch in chunks:
        content = ch["content"]
        entities = ch.get("entities") or {}

        # Compute embedding
        vec = embed_text(content)

        doc = {
            "id": ch["id"],
            "content": content,
            "doc_id": ch["doc_id"],
            "page_number": ch.get("page_number", None),
            "section_title": ch["section_title"],
            "tags": ch.get("tags", []),
            "contract_id": entities.get("contract_id"),
            "tenant_name": entities.get("tenant_name"),
            "owner_name": entities.get("owner_name"),
            "property_location": entities.get("property_location"),
            "gla": entities.get("gla"),
            "lease_amount": entities.get("lease_amount"),
            "rent_per_sqft": entities.get("rent_per_sqft"),
            "start_date": entities.get("start_date"),
            "end_date": entities.get("end_date"),
            "embedding": vec,
        }
        docs.append(doc)

    if not docs:
        print("No docs to index.")
        return

    print(f"Uploading {len(docs)} documents to Azure Search…")
    result = search_client.upload_documents(docs)
    failed = [r for r in result if not r.succeeded]
    if failed:
        print(f"Some documents failed to index: {failed}")
    else:
        print("All documents indexed successfully.")


def delete_all_documents():
    """Deletes all documents in the Azure Cognitive Search index."""
    print("🗑 Deleting all documents from Azure Search...")

    try:
        # Query all documents
        results = search_client.search(search_text="*", top=1000)
        ids = [d["id"] for d in results]

        if not ids:
            print("✔ Index already empty.")
            return

        # Build deletion payload
        actions = [{"@search.action": "delete", "id": _id} for _id in ids]

        # Push delete request
        search_client.upload_documents(actions)
        print(f"✔ Deleted {len(ids)} documents from index.")

    except Exception as e:
        print("❌ Error flushing index:", e)


    def generate_final_answer(query, chunks):
        # Build context text
        context = "\n\n".join(
            f"[Chunk {i+1}] (score={c['score']:.3f}) {c['content']}"
            for i, c in enumerate(chunks)
        )

        # Build citation metadata separately
        citations = []
        for i, c in enumerate(chunks, start=1):
            citations.append(
                f"[{i}] {c.get('filename', 'unknown-file')} "
                f"— chunk {c.get('id')} — score {c['score']:.3f}"
            )

        citation_text = "\n".join(citations)

        # LLM prompt
        prompt = f"""
                You are a UAE commercial lease expert.

                Use ONLY the provided contract chunks to answer the user's question.

                User Question:
                {query}

                Context Chunks:
                {context}

                After answering, include a "SOURCES:" section listing which chunks you used.
                Use this exact format:

                SOURCES:
                [number] filename — chunk id — score

                Now provide the best possible answer.
            """

        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )

        final_answer = response.choices[0].message.content

        # Append sources at bottom
        final_answer += "\n\nSOURCES:\n" + citation_text

        return final_answer
