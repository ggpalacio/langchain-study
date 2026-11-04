"""ChatPromptTemplate: dynamic prompts made of role-tagged messages.

Unlike PromptTemplate (a single string), ChatPromptTemplate builds a list
of messages with roles - "system" (global instructions/behavior) and
"user"/"human" (the actual question) - which is the format chat models
expect natively.

Docs: https://docs.langchain.com/oss/python/langchain/messages
Reference: https://reference.langchain.com/python/langchain-core/prompts/chat
Notion ("Primeiros Passos" - Aula 4): PromptTemplate e ChatPromptTemplate
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Each tuple is (role, template_string); both accept placeholders.
system = ("system", "you are an assistant that answers question in {style} style")
user = ("user", "{question}")

chat_prompt_template = ChatPromptTemplate([system, user])

# format_messages() fills the placeholders and returns a list of Message
# objects (SystemMessage, HumanMessage, ...) ready to send to a chat model.
messages = chat_prompt_template.format_messages(
    style = "few words",
    question = "What ChatPromptTemplate (LangChain) does exactly?"
)
for message in messages:
    print(f"{message.type}: {message.content}")

model = ChatAnthropic(model="claude-sonnet-4-6")
result = model.invoke(messages)
print(result.content)