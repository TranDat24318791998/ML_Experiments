from dotenv import load_dotenv

# Load LANGSMITH_* vars from the repo-root .env before any LangChain import reads them
load_dotenv()

from langsmith import traceable
from langchain.agents import create_agent
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain.chat_models import init_chat_model

_SYSTEM_PROMPT = "You are a helpful assitant"

llm = init_chat_model(
    model="qwen3.5:9b",
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0,
    reasoning=True,  # capture thinking into additional_kwargs["reasoning_content"]
    output_version="v1",  # put reasoning into message content blocks so LangSmith renders it
)

# Tools
def say_hi(name:str) -> str:
    """Say hi with a person"""
    return f"Hi {name}, how are you doing today?"

def get_weather(city:str) -> str:
    """Get weather information in a city"""
    return f"The weather in {city} is sunny!!!"

agent = create_agent(
    model=llm,
    system_prompt=_SYSTEM_PROMPT,
    tools=[say_hi, get_weather],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My name is Dat, let say hi"
            }
        ]
    }
)

ans = result['messages'][-1].text
print(ans)

# Traces are sent in a background thread; flush them before the script exits
wait_for_all_tracers()