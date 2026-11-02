"""Same ReAct agent as agent_tools.py, but pulling the prompt from a Hub.

This is the same calculator / mock-search ReAct agent as agent_tools.py
(see that file for the loop/tool-decorator explanations) - the only
difference is where the ReAct prompt comes from: instead of writing out
the Thought/Action/Observation instructions by hand, it's downloaded
ready-made from LangChain Hub via the LangSmith SDK. That's the
`LangChain Hub` mentioned in the course notes as a catalog for
publishing, versioning and pulling prompts (now backed by LangSmith).

Docs: https://docs.langchain.com/oss/python/langchain/agents
Hub prompt used: https://smith.langchain.com/hub/hwchase17/react
Notion ("Introdução ao LangChain" - Aula 2): LangChain Hub
"""
from langchain_classic.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client
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

# LangSmith's Client doubles as the Hub client: pull_prompt() downloads
# a ready-made prompt by "<owner>/<name>" instead of writing one by hand
# (this is the same hwchase17/react template agent_tools.py defines
# inline). dangerously_pull_public_prompt=True is required for prompts
# you don't own - it's a safety flag acknowledging you're trusting
# someone else's prompt text, which is executed as-is against your LLM.
client = Client()
prompt_template = client.pull_prompt(
    prompt_identifier="hwchase17/react",
    dangerously_pull_public_prompt=True)

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