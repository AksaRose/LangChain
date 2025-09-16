import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START,MessagesState,StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import SystemMessage, trim_messages

load_dotenv()

print("Tracing:",os.getenv("LANGSMITH_TRACING"))
print("API key:",os.getenv("LANGSMITH_API_KEY"))
print("Google API key:",os.getenv("GOOGLE_API_KEY"))

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Define a new graph
workflow = StateGraph(state_schema=State)

from langchain.chat_models import init_chat_model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Define the function that calls the model
def call_model(state: State):
    print(f"Messages before trimming: {len(state['messages'])}")
    trimmed_messages = trimmer.invoke(state["messages"])
    print(f"Messages after trimming: {len(trimmed_messages)}")
    print("Remaining messages:")
    for msg in trimmed_messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",

        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = messages + [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="|")
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()  # output contains all messages in state
