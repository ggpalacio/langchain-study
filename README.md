# langchain-study

Personal study project for experimenting with [LangChain](https://docs.langchain.com/) and its building blocks (prompts, chains, runnables, output parsers), using Anthropic's Claude models. These are small, self-contained example scripts, not a production application.

This project follows the "Introdução ao LangChain" module of the MBA Engenharia de Software com IA, whose class notes (in Portuguese) are here: [LangChain - Notion notes](https://app.notion.com/p/409934db1be4818b894fd9ad15e7ed6f).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Anthropic API key:

```bash
cp .env.example .env
```

The `doc_loader/` examples below load and split documents only - they don't
need a vector store. `rag/store.py` and `rag/search.py` do: start it with
`docker compose up -d` (Postgres + pgvector) before running either of
them, and fill in `PGVECTOR_URL` and `PGVECTOR_COLLECTION` in `.env`. The
embedding model they use (`HuggingFaceEmbeddings`) runs locally - no API
key needed for that part, `HF_TOKEN` is only for Hugging Face models/datasets
that require authentication.

## Examples

Each script is runnable on its own (`python <file>.py`) and focuses on a single LangChain concept:

| File | Concept | Docs |
| --- | --- | --- |
| [chat_anthropic.py](chat/chat_anthropic.py) | Hello World: a single call to `ChatAnthropic` | [Anthropic integration](https://docs.langchain.com/oss/python/integrations/chat/anthropic) |
| [init_chat_model.py](chat/init_chat_model.py) | `init_chat_model`: provider-agnostic model factory | [Models](https://docs.langchain.com/oss/python/langchain/models) |
| [prompt_template.py](prompt/prompt_template.py) | `PromptTemplate`: plain-text prompts with placeholders | [Messages](https://docs.langchain.com/oss/python/langchain/messages) |
| [chat_prompt_template.py](prompt/chat_prompt_template.py) | `ChatPromptTemplate`: role-based (system/user) prompts | [Messages](https://docs.langchain.com/oss/python/langchain/messages) |
| [chain.py](chain/chain.py) | LCEL: piping a prompt into a model with `\|` | [Philosophy](https://docs.langchain.com/oss/python/langchain/philosophy) |
| [chain_decorator.py](chain/chain_decorator.py) | `@chain`: turning a function into a `Runnable` | [chain decorator](https://reference.langchain.com/python/langchain-core/runnables/base/chain) |
| [runnable_lambda.py](chain/runnable_lambda.py) | `RunnableLambda`: wrapping an existing function | [RunnableLambda](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda) |
| [output_parsers.py](chain/output_parsers.py) | `StrOutputParser` + a multi-step translate-then-summarize pipeline | [StrOutputParser](https://reference.langchain.com/python/langchain-core/output_parsers/StrOutputParser) |
| [summarize.py](summarize/summarize.py) | Summarization, `"stuff"` strategy: all chunks in one prompt (also notes the `"map_reduce"` option) | [Summarization chains](https://reference.langchain.com/python/integrations/langchain_classic/chains/summarize) |
| [summarize_map_reduce.py](summarize/summarize_map_reduce.py) | `"map_reduce"` strategy built by hand with LCEL: summarize chunks, then combine | [Runnable](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable) |
| [agent_tools.py](agents/agent_tools.py) | ReAct agent (legacy API) that picks between a calculator and a mock search tool | [Agents](https://docs.langchain.com/oss/python/langchain/agents) · [Tools](https://docs.langchain.com/oss/python/langchain/tools) |
| [agent_tools_using_hub.py](agents/agent_tools_using_hub.py) | Same agent, pulling its ReAct prompt from LangChain Hub instead of writing it by hand | [Hub prompt](https://smith.langchain.com/hub/hwchase17/react) |
| [memory_history.py](history/memory_history.py) | `RunnableWithMessageHistory`: per-session chat memory across turns | [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) |
| [sliding_window_history.py](history/sliding_window_history.py) | Same idea, bounded with `trim_messages`: only the most recent turns reach the prompt | [trim_messages](https://reference.langchain.com/python/langchain-core/messages/trim_messages) |
| [pdf_loader.py](doc_loader/pdf_loader.py) | `PyPDFLoader` + `RecursiveCharacterTextSplitter`: load a local PDF and split it into overlapping chunks | [Document loaders](https://docs.langchain.com/oss/python/langchain/retrieval#document-loaders) |
| [web_base_loader.py](doc_loader/web_base_loader.py) | Same splitting pattern, loading a web page with `WebBaseLoader` instead of a file | [WebBaseLoader](https://reference.langchain.com/python/integrations/langchain_community/document_loaders/WebBaseLoader) |
| [store.py](rag/store.py) | RAG, step 1: embed a PDF's chunks (local `HuggingFaceEmbeddings`) and store them in pgvector | [Vector stores](https://docs.langchain.com/oss/python/langchain/retrieval#vector-stores) |
| [search.py](rag/search.py) | RAG, step 2: similarity search over that pgvector collection, scored by distance | [PGVector](https://reference.langchain.com/python/integrations/langchain_postgres/vectorstores/PGVector) |

## Reference

- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain Expression Language (LCEL) / Runnables](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable)
- [MBA Engenharia de Software com IA - LangChain module notes](https://app.notion.com/p/409934db1be4818b894fd9ad15e7ed6f)
