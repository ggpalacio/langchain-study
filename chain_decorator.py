"""@chain decorator: turning a plain function into a Runnable.

Business logic usually lives in regular functions. The `@chain` decorator
(imported from langchain_core.runnables) wraps a function so it gains
`invoke()` and can be piped with `|` just like a prompt or a model.
Golden rule: when a function feeds a PromptTemplate, it must return a
dict whose keys match the template's expected variables.

Reference: https://reference.langchain.com/python/langchain-core/runnables/base/chain
Notion ("Chains" - Aula 2): Chains como Decorators
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()

@chain
def square(number:int) -> dict:
    # Returns a dict because the next step (template) expects the
    # "squared_number" variable as a dict key.
    return {"squared_number": number*number}

template = PromptTemplate(
    input_variables=["squared_number"],
    template="Tell me some curiosity about number {squared_number}"
)

model = ChatAnthropic(model="claude-sonnet-4-6")
# Note: this rebinds the name `chain` from the decorator above to the
# pipeline itself - works, but shadowing the import like this is only
# fine in a short study script.
chain = square | template | model

result = chain.invoke(3)
print(result.content)