"""
LLM 客户端初始化 — 类比 JS 的 apiClient.js
整个项目只需要在这里配置一次，其他文件 from model import llm 即可使用。
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env（类比 JS 的 dotenv.config()）
load_dotenv(override=True)

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model="gpt-5.2",
    temperature=0.3,
    # 代理服务需要清空 SDK 特征 headers，否则会被拦截
    default_headers={
        "User-Agent": "",
        "x-stainless-lang": "",
        "x-stainless-package-version": "",
        "x-stainless-os": "",
        "x-stainless-arch": "",
        "x-stainless-runtime": "",
        "x-stainless-runtime-version": "",
        "x-stainless-async": "",
        "x-stainless-retry-count": "",
        "x-stainless-read-timeout": "",
    },
)
