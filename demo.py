import os
import requests
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv(override=True)

# ==========================
# 1️⃣ 初始化模型（必须开启streaming）
# ==========================

model = ChatOpenAI(
    api_key=os.getenv("LONGCAT_API_KEY"),
    base_url=os.getenv("LONGCAT_BASE_URL"),
    model="LongCat-Flash-Thinking-2601",
    temperature=0.3,
    streaming=True,  # 🔥关键
)

# ==========================
# 2️⃣ 定义工具
# ==========================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url, timeout=10)
    data = response.json()
    current = data["current_condition"][0]
    return f"{city}当前天气：{current['weatherDesc'][0]['value']}，气温{current['temp_C']}摄氏度"


@tool
def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气推荐旅游景点"""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"{city} 在 {weather} 天气下适合去的旅游景点推荐及理由"
    response = tavily.search(query=query, include_answer=True)
    return response.get("answer", "未找到相关信息")


tools = [get_weather, get_attraction]

# ==========================
# 3️⃣ 创建 Agent
# ==========================

agent = create_agent(
    model=model,
    tools=tools,
)

# ==========================
# 4️⃣ 流式执行
# ==========================

if __name__ == "__main__":

    stream = agent.stream(
        {"messages": [("user", "我在成都，请你给我推荐去哪玩")]},
        stream_mode="values"   # 🔥 关键
    )

    print("\n====== 流式输出 ======\n")

    for chunk in stream:
        message = chunk["messages"][-1]
        if message.content:
            print(message.content, end="", flush=True)

    print("\n\n====== 结束 ======")