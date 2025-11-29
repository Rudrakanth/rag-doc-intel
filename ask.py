from src.query.search_query import answer_with_search

q = "What are set off clauses?"
result = answer_with_search(q)
print("\nANSWER:\n", result["answer"])
