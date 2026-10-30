"""Chains & LCEL: piping a prompt straight into a model.

A chain connects Runnables with the `|` operator (LCEL - LangChain
Expression Language), so the output of one step becomes the input of the
next. Here `question | model` builds a chain that formats the prompt and
calls the LLM in a single `invoke()`, instead of doing it in two steps.
The resulting chain is itself a Runnable.

Docs: https://docs.langchain.com/oss/python/langchain/philosophy
Reference: https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence
Notion ("Chains" - Aula 1): Iniciando com Chains
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

question = PromptTemplate(
    input_variables=["name"],
    template="Hi, I am {name}! Tell me a joke with my name!"
)

model = ChatAnthropic(model="claude-sonnet-4-6")

# LCEL pipe: question's formatted output becomes the model's input.
chain = question | model
result = chain.invoke({"name": "Gabriel"})
print(result.content)