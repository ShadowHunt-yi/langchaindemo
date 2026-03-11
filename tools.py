"""
工具函数 — 类比 JS 的 services/ 目录
每个函数是一个独立的工具，被 agent 调用。
"""
import os
import requests
from tavily import TavilyClient


def get_weather(city: str) -> str:
    """通过 wttr.in API 查询实时天气。"""
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        desc = current["weatherDesc"][0]["value"]
        temp = current["temp_C"]
        return f"{city}当前天气:{desc}，气温{temp}摄氏度"
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气，使用 Tavily Search API 搜索推荐的旅游景点。"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return response["answer"]

        results = [
            f"- {r['title']}: {r['content']}" for r in response.get("results", [])
        ]
        if not results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        return "根据搜索，为您找到以下信息:\n" + "\n".join(results)
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


# 工具注册表 — 类比 JS 的 { weatherService, attractionService }
# agent.py 通过这个 dict 按名字查找并调用工具
TOOLS = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
