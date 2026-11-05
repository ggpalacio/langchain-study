"""Document loading, part 1: turning a local PDF into retrievable chunks.

PyPDFLoader reads a PDF one Document per page (page_content + metadata like
page number). RecursiveCharacterTextSplitter then re-chunks that into
smaller, overlapping pieces sized for embedding/retrieval: chunk_overlap
keeps a slice of the previous chunk's tail in the next chunk, so a fact
split across a chunk boundary still appears intact in at least one chunk.

This script only loads and splits - see web_base_loader.py for the same
pattern against a web page instead of a file. Embedding/storing the chunks
(e.g. into the pgvector instance from docker-compose.yaml) is a later step.

Docs: https://docs.langchain.com/oss/python/langchain/retrieval#document-loaders
Reference: https://reference.langchain.com/python/integrations/langchain_community/document_loaders/PyPDFLoader
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Resolved relative to this file (not the current working directory), so
# the script still finds the PDF when run from the repo root as
# `python doc_loader/pdf_loader.py`.
pdf_path = Path(__file__).parent / "Claude Opus 5.5 System Card.pdf"

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(len(chunks))