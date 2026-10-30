"""RunnableLambda: wrapping a function you can't (or shouldn't) decorate.

@chain works great for functions you own, but sometimes the function
comes from third-party code, a legacy module, or just shouldn't be
modified. RunnableLambda wraps any callable from the outside, turning
it into a Runnable ready to be piped with `|`, without touching its
definition.

Reference: https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda
Notion ("Chains" - Aula 3): Runnable Lambdas
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def square(number:int) -> dict:
    # Plain function, unaware of LangChain - RunnableLambda is what
    # makes it chainable below.
    return {"squared_number": number*number}

runnable = RunnableLambda(square)

template = PromptTemplate(
    input_variables=["squared_number"],
    template="Tell me some curiosity about number {squared_number}"
)

model = ChatAnthropic(model="claude-sonnet-4-6")
chain = runnable | template | model

result = chain.invoke(4)
print(result.content)