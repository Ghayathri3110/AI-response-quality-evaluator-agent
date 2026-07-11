from sentence_transformers import SentenceTransformer
import chromadb

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Connect to ChromaDB
# -----------------------------
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# -----------------------------
# Load collection
# -----------------------------
collection = client.get_collection(
    name="knowledge_base"
)


def retrieve_context(question, top_k=3):
    """
    Retrieve the most relevant context chunks.
    """

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    question = input("Enter a question: ")

    contexts = retrieve_context(question)

    print("\nRetrieved Context:\n")

    for i, context in enumerate(contexts, 1):
        print(f"{i}. {context}\n")