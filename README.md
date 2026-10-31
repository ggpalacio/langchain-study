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

## Examples

Each script is runnable on its own (`python <file>.py`) and focuses on a single LangChain concept:

| File | Concept | Docs |
| --- | --- | --- |
| [chat_anthropic.py](chat_anthropic.py) | Hello World: a single call to `ChatAnthropic` | [Anthropic integration](https://docs.langchain.com/oss/python/integrations/chat/anthropic) |
| [init_chat_model.py](init_chat_model.py) | `init_chat_model`: provider-agnostic model factory | [Models](https://docs.langchain.com/oss/python/langchain/models) |
| [prompt_template.py](prompt_template.py) | `PromptTemplate`: plain-text prompts with placeholders | [Messages](https://docs.langchain.com/oss/python/langchain/messages) |
| [chat_prompt_template.py](chat_prompt_template.py) | `ChatPromptTemplate`: role-based (system/user) prompts | [Messages](https://docs.langchain.com/oss/python/langchain/messages) |
| [chain.py](chain.py) | LCEL: piping a prompt into a model with `\|` | [Philosophy](https://docs.langchain.com/oss/python/langchain/philosophy) |
| [chain_decorator.py](chain_decorator.py) | `@chain`: turning a function into a `Runnable` | [chain decorator](https://reference.langchain.com/python/langchain-core/runnables/base/chain) |
| [runnable_lambda.py](runnable_lambda.py) | `RunnableLambda`: wrapping an existing function | [RunnableLambda](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda) |
| [output_parsers.py](output_parsers.py) | `StrOutputParser` + a multi-step translate-then-summarize pipeline | [StrOutputParser](https://reference.langchain.com/python/langchain-core/output_parsers/StrOutputParser) |
| [summarize_stuff.py](summarize_stuff.py) | Summarization, `"stuff"` strategy: all chunks in one prompt | [Summarization chains](https://reference.langchain.com/python/integrations/langchain_classic/chains/summarize) |
| [summarize_map_reduce.py](summarize_map_reduce.py) | Summarization, `"map_reduce"` strategy: summarize chunks, then combine | [Text splitters](https://docs.langchain.com/oss/python/integrations/splitters/index) |

## Reference

- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain Expression Language (LCEL) / Runnables](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable)
- [MBA Engenharia de Software com IA - LangChain module notes](https://app.notion.com/p/409934db1be4818b894fd9ad15e7ed6f)
