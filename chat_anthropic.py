"""Hello World: first call to an LLM through LangChain.

`ChatAnthropic` wraps Anthropic's chat models behind LangChain's common
chat model interface, so swapping providers later (OpenAI, Gemini, ...)
would only mean changing this class, not the rest of the pipeline.

Docs: https://docs.langchain.com/oss/python/integrations/chat/anthropic
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Loads ANTHROPIC_API_KEY (and friends) from .env into the environment.
load_dotenv()

model = ChatAnthropic(model="claude-sonnet-4-6")

# invoke() sends the prompt and returns an AIMessage; .content holds the text.
message = model.invoke("Which are available Anthropic models to use in LangChain?")
print(message.content)