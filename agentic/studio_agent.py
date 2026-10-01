"""Agent graph exposed to LangSmith Studio via langgraph.json.

Only defines `agent` — Studio loads this module and runs the graph itself,
so nothing here should call `agent.invoke(...)` at import time.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

_SYSTEM_PROMPT = "You are a helpful assitant"

llm = init_chat_model(
    model="qwen2.5:14b",
    model_provider="ollama",
    base_url="http://localhost:11434",
    temperature=0,
    output_version="v1",
)


# Tools
@tool
def say_hi(name: str) -> str:
    """Say hi with a person"""
    return f"Hi {name}, how are you doing today?"


@tool
def give_advice() -> str:
    """Give a person advice"""
    return "You tried your best, so keep pushing, keep trying"


@tool
def get_weather(city: str) -> str:
    """Get weather information in a city"""
    return f"The weather in {city} is sunny!!!"


agent = create_agent(
    model=llm,
    system_prompt=_SYSTEM_PROMPT,
    tools=[say_hi, get_weather, give_advice],
)
