"""
Agent 循环 — 类比 JS 的 controller / middleware

核心流程（ReAct 模式）:
  用户提问 → LLM 思考 → 输出 Action
    ├─ Action 是工具调用 → 执行工具 → 把结果喂回 LLM → 继续循环
    └─ Action 是 Finish[答案] → 输出最终答案 → 结束

你之前的代码只做了"LLM 思考"这一步，没有真正执行工具，
所以 LLM 只能瞎猜，这里我们补上完整循环。
"""
import re
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from model import llm
from tools import TOOLS

# 读取 prompt 文件 — Path 类似 JS 的 path.join(__dirname, ...)
PROMPT_FILE = Path(__file__).parent / "prompts" / "travel_agent.txt"
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

# 最大循环次数，防止死循环（类比 JS 的 for 循环 + break）
MAX_ITERATIONS = 10


def _parse_action(text: str):
    """
    从 LLM 输出中解析 Action 行。
    返回 (action_type, action_content)：
      - ("tool", "get_weather(city=\"成都\")")
      - ("finish", "最终答案内容")
      - (None, None) 表示解析失败
    """
    # 先尝试匹配多行 Finish：Action: Finish[...内容可能跨行...]
    # re.DOTALL 让 . 匹配换行符（类比 JS 的 /s 标志）
    finish_match = re.search(r"Action:\s*Finish\[(.+)\]", text, re.DOTALL)
    if finish_match:
        return "finish", finish_match.group(1).strip()

    # 再匹配单行工具调用：Action: get_weather(city="成都")
    match = re.search(r"Action:\s*(.+)", text)
    if not match:
        return None, None

    return "tool", match.group(1).strip()


def _execute_tool(action_str: str) -> str:
    """
    解析并执行工具调用字符串，如 get_weather(city="成都")
    类比 JS: const fn = tools[funcName]; fn(...args)
    """
    # 匹配函数名和参数部分
    match = re.match(r"(\w+)\((.+)\)", action_str)
    if not match:
        return f"错误:无法解析工具调用 '{action_str}'"

    func_name = match.group(1)
    args_str = match.group(2)

    # 查找工具
    func = TOOLS.get(func_name)
    if not func:
        return f"错误:未知工具 '{func_name}'，可用工具: {list(TOOLS.keys())}"

    # 解析参数：提取所有 key="value" 对
    kwargs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_str))

    try:
        return func(**kwargs)
    except TypeError as e:
        return f"错误:调用 {func_name} 参数不匹配 - {e}"


def run(user_input: str):
    """
    运行 Agent 循环。
    类比 JS 的 async function runAgent(input) { while (true) { ... } }
    """
    # messages 列表 — 类似 JS 数组，存放对话历史
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    # 记录是否调用过工具（防止 LLM 偷懒直接 Finish）
    tool_called = False

    for i in range(MAX_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"第 {i + 1} 轮")
        print("=" * 50)

        # 流式输出 — 类比 JS 的 for await (const chunk of stream)
        # 边打印边收集，收集完再解析 Action
        # stop=["Observation:"] 让 LLM 输出完一对 Thought+Action 后自动停下
        # 原理：LLM 知道 Action 之后应该是 Observation，看到 stop 词就停嘴等结果
        reply = ""
        for chunk in llm.stream(messages, stop=["Observation:"]):
            print(chunk.content, end="", flush=True)
            reply += chunk.content
        print()  # 流结束后换行

        # 手动构造 AIMessage 放入历史（stream 不返回完整 message 对象）
        response = AIMessage(content=reply)

        # 解析 Action
        action_type, action_content = _parse_action(reply)

        if action_type == "finish":
            if not tool_called:
                # 还没调过工具就想 Finish？打回去重来
                messages.append(response)
                messages.append(HumanMessage(
                    content="你还没有调用任何工具。请先使用 get_weather 查询天气，再继续。"
                ))
                continue

            # 到终点了，输出最终答案
            print(f"\n{'='*50}")
            print("最终答案")
            print("=" * 50)
            # 处理 \n 字面量 → 真实换行
            answer = action_content.replace("\\n", "\n")
            print(answer)
            return answer

        if action_type == "tool":
            # 执行工具，把结果作为 Observation 喂回 LLM
            print(f"\n→ 执行工具: {action_content}")
            result = _execute_tool(action_content)
            print(f"← 工具返回: {result[:200]}...")  # 只预览前200字
            tool_called = True

            # 把 LLM 的回复和工具结果都加入对话历史
            messages.append(response)  # LLM 的 Thought+Action
            messages.append(HumanMessage(content=f"Observation: {result}"))
        else:
            # 解析失败，提示 LLM 重新输出
            messages.append(response)
            messages.append(HumanMessage(content="请严格按照格式输出 Thought 和 Action。"))

    print("达到最大循环次数，Agent 停止。")
