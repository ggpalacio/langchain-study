"""Short-term memory: giving a chain access to past turns of a conversation.

A plain chain is stateless - each invoke() starts from scratch. To hold
a conversation, past messages need to be fed back into the prompt on
every call. RunnableWithMessageHistory automates that: it wraps a chain,
reads the right chat history for the current session before calling it,
and appends the new human/AI messages to that history afterwards.

NOTE: RunnableWithMessageHistory is deprecated upstream (LangChain now
recommends LangGraph's built-in persistence for this), but it's kept
here because it demonstrates the memory concept directly without
requiring a LangGraph graph/checkpointer setup.

Docs: https://docs.langchain.com/oss/python/langchain/short-term-memory
Reference: https://reference.langchain.com/python/langchain-core/runnables/history/RunnableWithMessageHistory
Notion ("Introdução ao LangChain" - Aula 3): Memória
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

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

chain = prompt | chat_model

# One InMemoryChatMessageHistory per session_id, so unrelated
# conversations never see each other's messages. This dict only lives
# as long as the process runs - nothing is persisted to disk.
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# input_messages_key/history_messages_key must match the prompt's
# placeholders above ("input" and "history") so RunnableWithMessageHistory
# knows where to inject the new message and the accumulated history.
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
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

# A conversation that only makes sense with memory: the name given in
# the first message is recalled in the second, and the age question is
# only answerable after the birth date and "today" show up later -
# proving the model is reading the accumulated history, not just the
# latest message.
chat(conversational_chain, "Hello, my name is Gabriel, how are you?", config)
chat(conversational_chain, "Can you repeat my name?", config)
chat(conversational_chain, "Do you know how many years I am?", config)
chat(conversational_chain, "I was born in 1984, Apr 8th!", config)
chat(conversational_chain, "We are in 2026, Nov 3rd.", config)