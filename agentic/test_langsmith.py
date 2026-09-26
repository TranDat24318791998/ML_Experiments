from langsmith import traceable
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

_SYSTEM_PROMPT = "You are a helpful assitant"

llm = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0
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

ans = result['messages'][-1].content
print(ans)