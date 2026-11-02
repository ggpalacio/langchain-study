"""ReAct agent with tools (legacy API): the agent decides what to call.

Unlike a chain (fixed sequence of steps), an agent is given a set of
tools and a goal, and the LLM itself decides which tool to call, with
what input, and when it has enough information for a final answer. This
example implements the classic ReAct loop (Reason + Act): the model
alternates between "Thought" (reasoning), "Action"/"Action Input"
(calling a tool) and "Observation" (the tool's result) until it reaches
a "Final Answer".

create_react_agent + AgentExecutor live in langchain_classic - this is
the pre-v1 agent API, kept for chains/agents written before the
rewrite. LangChain v1's current default is create_agent (from
`langchain.agents`), which runs a similar loop but through middleware
(memory, summarization, guardrails...) instead of a hand-written prompt
like the one below.

Docs: https://docs.langchain.com/oss/python/langchain/agents
Tools: https://docs.langchain.com/oss/python/langchain/tools
Notion ("Introdução ao LangChain" - Aula 3): Agentes e ferramentas
"""
from langchain_classic.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# @tool turns a plain function into a Tool the agent can call. The
# function's docstring becomes the tool's description - it's the only
# thing the LLM reads to decide *when* this tool is the right one to use,
# so it needs to be clear even though it's never printed anywhere.
@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluates a simple mathematical expression and returns the result"""
    try:
        # NOTE: eval() runs arbitrary Python - fine for a local study
        # example where only the LLM supplies the expression, but never
        # do this with input that could come from an untrusted user.
        return eval(expression)
    except Exception as e:
        return str(e)


# A second tool with intentionally limited knowledge (a hardcoded dict
# instead of a real search/API call), so the example can show the model
# admitting "I don't know" instead of making up an answer when the tool
# can't help - that's what the prompt's "NEVER returns a final answer
# using data not returned by provided tools" rule is enforcing below.
@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in its mocked data."""
    capitals = {
        "Brazil": "Brasília",
        "United States": "Washington",
        "Spain": "Madrid",
        "England": "London",
        "France": "Paris",
        "Germany": "Berlin",
    }
    for country, capital in capitals.items():
        if country.lower() == query.lower():
            return f"The capital of {country} is: {capital}"

    return f"I don't know the capital of {query}."

# disable_streaming=True: the agent parses the model's full response as
# text (looking for "Action:", "Observation:", etc.), so a complete
# response is simpler to work with here than a token-by-token stream.
model = ChatAnthropic(model="claude-sonnet-4-6", disable_streaming=True)
tools = [calculator, web_search_mock]

# The classic ReAct prompt: {tools} and {tool_names} are filled in
# automatically by create_react_agent from the `tools` list above;
# {input} is the question passed to invoke(); {agent_scratchpad} is
# where the agent's own Thought/Action/Observation history accumulates
# across loop iterations, so the model can see what it already tried.
prompt_template = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools.
Only use the information you get from the tools, even if you know the answer.
NEVER returns a final answer using data not returned by provided tools. 
If the information is not provided by the tools, say you don't know.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for Observation.
- Never search the internet. Only use the tools provided.
- ONLY use data obtained in tools in your Final Answer.

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# create_react_agent wires the model, tools and prompt into a Runnable
# that produces one ReAct step (Thought/Action/Action Input) per call.
# stop_sequence=False skips the default "\nObservation" stop sequence,
# letting the model generate freely instead of being cut off there.
agent_chain = create_react_agent(model, tools, prompt_template, stop_sequence=False)

# AgentExecutor is what actually RUNS the loop: call the agent, parse
# its output, execute the requested tool, feed the Observation back in
# as part of agent_scratchpad, and repeat until a Final Answer appears
# or max_iterations is hit.
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain,
    tools=tools,
    verbose=True,  # prints every Thought/Action/Observation step
    handle_parsing_errors="Invalid format. Either provide an Action with Action Input or a Final Answer.",
    max_iterations=3)

# Two questions the mock tool can answer, one for the calculator tool,
# and a last one no tool covers - that last call is the actual test of
# the "say you don't know" rule instead of the model guessing "1 cm".
print(agent_executor.invoke({"input": "What is the capital of Spain?"}))
print(agent_executor.invoke({"input": "What is the capital of Portugal?"}))
print(agent_executor.invoke({"input": "How much is 10 + 10?"}))
print(agent_executor.invoke({"input": "How much is 1 centimeter?"}))