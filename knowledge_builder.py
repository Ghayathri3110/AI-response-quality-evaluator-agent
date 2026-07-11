from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import os
import shutil

# =====================================================
# Load SQuAD Dataset
# =====================================================

print("=" * 50)
print("Loading SQuAD dataset...")

squad = load_dataset(
    "rajpurkar/squad",
    split="train"
)

contexts = list(set(squad["context"]))

print(f"Loaded {len(contexts)} unique SQuAD contexts.")

# =====================================================
# Load Wikipedia Dataset
# =====================================================

print("=" * 50)
print("Loading Wikipedia dataset...")

wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train[:1000]"      # Increase later if needed
)

contexts.extend(wiki["text"])

print(f"Loaded {len(wiki)} Wikipedia articles.")



# =====================================================
# Chunking
# =====================================================

print("=" * 50)
print("Chunking documents...")


def chunk_text(text, chunk_size=100):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunks.append(
            " ".join(words[i:i + chunk_size])
        )

    return chunks


all_chunks = []

for doc in contexts:

    all_chunks.extend(
        chunk_text(doc)
    )

print(f"Created {len(all_chunks)} chunks.")

# =====================================================
# Load Embedding Model
# =====================================================

print("=" * 50)
print("Loading Sentence Transformer...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================================
# Generate Embeddings
# =====================================================

print("=" * 50)
print("Generating embeddings...")

embeddings = model.encode(
    all_chunks,
    show_progress_bar=True,
    batch_size=64
)

# =====================================================
# Recreate ChromaDB
# =====================================================

print("=" * 50)
print("Creating ChromaDB...")

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="knowledge_base"
)

# =====================================================
# Store Embeddings
# =====================================================

print("=" * 50)
print("Saving embeddings...")

BATCH_SIZE = 5000

for i in range(0, len(all_chunks), BATCH_SIZE):

    batch_chunks = all_chunks[i:i + BATCH_SIZE]

    batch_embeddings = embeddings[i:i + BATCH_SIZE]

    collection.add(

        ids=[
            str(j)
            for j in range(i, i + len(batch_chunks))
        ],

        documents=batch_chunks,

        embeddings=batch_embeddings.tolist()

    )

    print(
        f"Stored {i + len(batch_chunks)} / {len(all_chunks)} chunks"
    )

# =====================================================
# Finished
# =====================================================

print("=" * 50)
print("Knowledge Base Created Successfully!")
print("=" * 50)

print(f"SQuAD contexts        : {len(set(squad['context']))}")
print(f"Wikipedia articles    : {len(wiki)}")
print(f"Total documents       : {len(contexts)}")
print(f"Total chunks          : {collection.count()}")

print("=" * 50)