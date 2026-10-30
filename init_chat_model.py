"""init_chat_model: a provider-agnostic factory for chat models.

Instead of importing each provider's class (ChatAnthropic, ChatOpenAI,
ChatGoogleGenerativeAI...), init_chat_model() takes a model name and
provider and instantiates the right integration for you, reading the
API key from the environment. Handy when the provider/model should be
swappable (e.g. via config) without touching the rest of the code.

Docs: https://docs.langchain.com/oss/python/langchain/models
Notion ("Primeiros Passos" - Aula 3): init_chat_model, uma factory para
qualquer provedor
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

claude = init_chat_model(model="claude-sonnet-4-6", model_provider="anthropic")
message = claude.invoke("What is the difference between 'init_chat_model' vs 'ChatAnthropic' in LangChain?")
print(message.content)