"""RAG, step 2: similarity search over the chunks store.py embedded.

Connects to the same pgvector collection (same embedding model,
collection name, and connection string) and embeds `query` the same way
the stored chunks were embedded, so the two vectors are comparable.
similarity_search_with_score returns the k nearest chunks together with
their distance score (lower = closer match for pgvector's default
cosine distance) instead of just the chunks, which is what makes it
possible to judge how confident a given match actually is.

Run store.py at least once first - this has nothing to search against
otherwise.

Docs: https://docs.langchain.com/oss/python/langchain/retrieval#vector-stores
Reference: https://reference.langchain.com/python/integrations/langchain_postgres/vectorstores/PGVector
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

query = "Tell me what is new in Opus 5.5?"

store = PGVector(
    embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    collection_name=os.getenv("PGVECTOR_COLLECTION", ""),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)
results = store.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(results):
    print("-"*50)
    print(f"{i}: {score:.2f}")
    print(f"Text: {doc.page_content}")
    print("Metadata")
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")
