from retrieval_agent import retrieve_context
print(retrieve_context("computer science", top_k=3))
print(retrieve_context("mathematics", top_k=3))
print(retrieve_context("history", top_k=3))