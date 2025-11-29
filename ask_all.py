import sys

from src.query.search_query import answer_with_search, search_contracts


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask_all.py "your question"')
        sys.exit(1)

    query = sys.argv[1]

    print("\n===============================")
    print("           RAG QUERY")
    print("===============================\n")

    print(f"User Question: {query}\n")

    print("Running vector + keyword search on DI index...")
    results = search_contracts(query, top_k=50)
    print(f"Results found: {len(results)}\n")

    print("===============================")
    print("          TOP MATCHES")
    print("===============================\n")

    for idx, r in enumerate(results, start=1):
        print("----")
        print(f"[{idx}] score={r['@search.score']:.3f} | file={r.get('filename')}")
        print(f"Chunk ID: {r['id']}")
        print(f"Content: {r['content'][:200]}...")
        print("----\n")

    print("\n===============================")
    print("      GENERATING ANSWER")
    print("===============================\n")

    answer = answer_with_search(query, chat_history=[])["answer"]

    print("\nANSWER:\n")
    print(answer)

    print("\n===============================")
    print("             DONE")
    print("===============================\n")


if __name__ == "__main__":
    main()
