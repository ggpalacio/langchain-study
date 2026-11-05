"""Document loading, part 2: turning a web page into retrievable chunks.

WebBaseLoader fetches the URL and strips it down to one Document holding
the page's text (via BeautifulSoup under the hood, hence the
beautifulsoup4 dependency). From there it's the same splitting step as
pdf_loader.py: RecursiveCharacterTextSplitter re-chunks that text with
overlap so facts near a chunk boundary aren't lost to either chunk alone.

Docs: https://docs.langchain.com/oss/python/langchain/retrieval#document-loaders
Reference: https://reference.langchain.com/python/integrations/langchain_community/document_loaders/WebBaseLoader
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://www.langchain.com")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
for chunk in chunks:
    print(chunk)
    print("-"*30)