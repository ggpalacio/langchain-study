"""Short-term memory with a sliding window: bounding how much history is kept.

memory_history.py feeds the *entire* session history into every prompt,
which eventually overflows the model's context window on a long enough
conversation. This script adds a `prepare` step (trim_messages) in
front of the prompt that keeps only the most recent messages, dropping
older ones instead of letting history grow forever - the classic
sliding-window tradeoff: bounded context size, at the cost of the model
eventually "forgetting" anything outside the window.

RunnableWithMessageHistory still owns the *full*, untrimmed history
(under the "raw_history" key here); `prepare_inputs` is what narrows it
down to a short "history" list right before it reaches the prompt. The
session store keeps growing either way - only what's shown to the model
is capped.

NOTE: RunnableWithMessageHistory is deprecated upstream (LangChain now
recommends LangGraph's built-in persistence for this), but it's kept
here because it demonstrates the memory concept directly without
requiring a LangGraph graph/checkpointer setup.

Docs: https://docs.langchain.com/oss/python/langchain/short-term-memory
Reference: https://reference.langchain.com/python/langchain-core/messages/trim_messages
Notion ("Introdução ao LangChain" - Aula 3): Memória
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda

load_dotenv()

# MessagesPlaceholder reserves a spot in the prompt for a *list* of
# messages (the conversation history) instead of a single placeholder
# value - RunnableWithMessageHistory fills it in on every call.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chat_model = ChatAnthropic(model="claude-sonnet-4-6")

# Narrows the full session history down to what actually gets sent to
# the model. token_counter=len treats each message as costing "1 token"
# (instead of a real tokenizer), so max_tokens=2 really means "keep
# about the last 2 messages" - deliberately tiny here so the window
# fills up and drops old messages after just one or two turns, making
# the chat() calls below visibly "forget" things.
def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])
    trimmed_history = trim_messages(
        raw_history,
        token_counter=len,
        max_tokens=2,
        strategy="last",       # keep the most recent messages, drop the oldest
        start_on="human",      # don't cut a trailing AI reply without its human turn
        include_system=True,   # always keep the system prompt regardless of budget
        allow_partial=False,
    )
    return {"input": payload.get("input", ""), "history": trimmed_history}

prepare = RunnableLambda(prepare_inputs)
# prepare receives {"input": ..., "raw_history": [...]} from
# RunnableWithMessageHistory below and remaps it to {"input": ...,
# "history": [...]} - the shape prompt's MessagesPlaceholder expects.
chain = prepare | prompt | chat_model

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# history_messages_key="raw_history" (not "history"!) is what makes this
# the *untrimmed* history: RunnableWithMessageHistory injects the full
# session log under that key, and `prepare` is what's responsible for
# turning it into the short "history" the prompt actually sees.
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history",
)

# session_id is how get_session_history picks which conversation this
# call belongs to - change it to start a brand new, empty history.
config = {"configurable": {"session_id": "demo-session"}}

def chat(chain, input, config):
    # Each call only needs to pass the new message: the chain fetches
    # and prepends the session's prior history automatically, then
    # appends both this human message and the AI's reply to it.
    response = chain.invoke(
        {"input": input},
        config=config,
    )
    print("Human:", input)
    print("Assistant:", response.content)
    print("-" * 30)

# With a window this small, the name given in the first message is
# already outside it by the third call - unlike memory_history.py,
# where nothing is ever dropped, this is expected to "forget" the name.
chat(conversational_chain, "Hello, my name is Gabriel! Do not mention my name!", config)
chat(conversational_chain, "I am 42 years old!", config)
chat(conversational_chain, "Now you can say my name!", config)