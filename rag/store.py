"""RAG, step 1: embed a PDF's chunks and store them in pgvector.

Same load-and-split pattern as doc_loader/pdf_loader.py, plus the two
steps that turn those chunks into a searchable vector store: embedding
(HuggingFaceEmbeddings, a local sentence-transformers model - no API
call) and storage (PGVector, langchain-postgres' wrapper around the
pgvector Postgres extension started by docker-compose.yaml). search.py
queries this same store once it's been populated.

Rerunning this script is safe: PGVector upserts by id, so the fixed
"doc-{i}" ids below just overwrite the same rows instead of duplicating
them, as long as the chunking is unchanged.

Docs: https://docs.langchain.com/oss/python/langchain/retrieval#vector-stores
Reference: https://reference.langchain.com/python/integrations/langchain_postgres/vectorstores/PGVector
"""

import os
import pathlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# read
# Resolved relative to this file, not the current working directory, so
# this still finds the PDF when run from the repo root.
current = pathlib.Path(__file__).parent.parent.absolute()
loader = PyPDFLoader(str(current) + "/doc_loader/Claude Opus 5.5 System Card.pdf")
doc = loader.load()

# split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
docs = splitter.split_documents(doc)

# enrich
# PyPDFLoader fills in metadata keys (e.g. "author") as "" when a PDF
# doesn't set them; dropping those empty/None values keeps the metadata
# PGVector stores per chunk limited to what's actually meaningful.
enriched_docs = [
    Document(
        page_content=doc.page_content,
        metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)},
    )
    for doc in docs
]

# embedding
ids = [f"doc-{i}" for i in range(len(enriched_docs))]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION", ""),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)
# ids must be passed as the `ids=` keyword - add_documents() only takes
# `documents` positionally, so `ids` as a second positional argument
# raises a TypeError.
store.add_documents(enriched_docs, ids=ids)



