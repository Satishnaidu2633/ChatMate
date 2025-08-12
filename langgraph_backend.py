from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is missing. Please set it in your .env file.")

# ✅ Initialize Groq LLM
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",  # Groq model
    api_key=GROQ_API_KEY
)

# Define state type
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Chat node function
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

# Build graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile chatbot
chatbot = graph.compile(checkpointer=checkpointer)
