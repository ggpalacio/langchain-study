"""PromptTemplate: dynamic prompts with placeholders.

A PromptTemplate holds a text template plus the variable names it expects,
so the same prompt can be reused with different inputs (from code, a file,
a database, etc.) instead of hardcoding strings everywhere.

Docs: https://docs.langchain.com/oss/python/langchain/messages
Notion ("Primeiros Passos" - Aula 4): PromptTemplate e ChatPromptTemplate
"""
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I am {name}! Tell me a joke with my name!"
)

# format() just fills in the placeholders and returns a plain string
# (no LLM call happens here).
text = template.format(name="Gabriel")
print(text)