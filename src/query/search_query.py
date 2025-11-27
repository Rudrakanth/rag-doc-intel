from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from src.config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_EMBED_ENDPOINT,
    AZURE_OPENAI_EMBED_DEPLOYMENT,
    openai_client
)

# ==========
# Clients
# ==========

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

aoai = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_EMBED_ENDPOINT,
)

# ==========
# Embeddings
# ==========

def embed_query(text: str):
    embedding = aoai.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBED_DEPLOYMENT
    )
    return embedding.data[0].embedding


# ==========
# Hybrid Vector Search
# ==========
def search_contracts(query: str, top_k: int = 5, filters: str = None):
    print("\n=== DEBUG: Running search_contracts ===")
    print("Query:", query)

    vector = embed_query(query)
    print("Embedding dim:", len(vector))

    vector_query = VectorizedQuery(
        vector=vector,
        fields="embedding",
        k_nearest_neighbors=top_k   
    )

    print("Sending query to Azure Search…")

    results = list(search_client.search(
        search_text=query,            # keyword
       # vector_queries=[vector_query], # vector
        filter=filters,
        top=top_k,        
    ))

    print("Results count:", len(results))
    for r in results:
        print("----")
        print("ID:", r["id"])
        print("Page:", r["page_number"])
        print("Content snippet:", r["content"][:200])
        print("Score:", r['@search.score'], 0.0)

    return results


system_prompt = """
You are an expert legal analyst specializing in UAE commercial lease agreements.

Your responsibilities:
1. Provide accurate, concise, contract-grounded answers.
2. Use ONLY the provided RAG chunks. 
3. If the answer is NOT in the chunks, say: "This information is not present in the retrieved contract text."
4. Cite every claim using the provided chunk IDs and page numbers.
5. DO NOT invent any legal terms, amounts, tenant names, dates, or clauses.
6. DO NOT summarize outside your context.
7. Keep answers clear, professional, and free from speculation.

Answer format:
-----------------------
[ANSWER]

[SOURCES]
1. filename — Page X — Chunk ID — Score
-----------------------
"""

# ==========
# RAG Answer Generator
# ==========
def generate_final_answer(query, chunks):
    """
    Use Azure OpenAI to generate a final answer with RAG citations.
    """

    if not chunks:
        return "No relevant information was found in the indexed documents."

    # Build context for RAG
    context_blocks = []
    citation_blocks = []

    for i, c in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Chunk {i}] (page={c.get('page_number')}) (score={c['@search.score']:.3f})\n{c['content']}"
        )
        citation_blocks.append(
            f"[{i}] {c.get('filename')} — Page {c.get('page_number')} — Chunk {c.get('id')} — Score {c['@search.score']:.3f}"
        )

    context_text = "\n\n".join(context_blocks)
    citation_text = "\n".join(citation_blocks)

    user_prompt = f"""
User question:
{query}

Here are the retrieved chunks. ONLY use information found inside them:

{context_text}

Now answer the user question strictly based on the above chunks.
"""

    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )

    final_answer = response.choices[0].message.content

    # Append citations at the end
    final_answer += "\n\nSOURCES:\n" + citation_text

    return final_answer