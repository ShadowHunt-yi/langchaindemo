"""
带记忆的 Agent 循环

架构说明（给 JS 开发者）：
  这个文件类比 JS 的 agentWithMemory.js
  它复用 agent.py 的解析/执行逻辑，新增记忆层：
    对话前 → 检索相关记忆，注入 system prompt
    对话中 → 正常 ReAct 循环（复用 agent.py 的核心函数）
    对话后 → 用 LLM 提取记忆并持久化

  类比 JS 的装饰器模式 / 中间件：
    agent.py      = 核心路由处理
    memory.py     = 数据库 service
    本文件         = 加了记忆中间件的路由
"""
import re
from pathlib import Path
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

# 复用已有模块 — 不再 Mock
from model import llm
from tools import TOOLS
from memory import get_memory, AgentMemory
from agent import _parse_action, _execute_tool

# 读取带记忆版的 prompt
PROMPT_FILE = Path(__file__).parent / "prompts" / "travel_agent_with_memory.txt"
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")

MAX_ITERATIONS = 10


def _extract_memories(messages: List[BaseMessage], memory: AgentMemory):
    """
    用 LLM 从对话中提取值得记住的信息，存入长期记忆。
    类比 JS: await extractAndSave(chatHistory, db)
    """
    # 只取最近 6 条消息，避免 token 太长
    recent = messages[-6:]
    conversation_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:300]}"
        for m in recent
        if not isinstance(m, SystemMessage)
    )

    extract_prompt = f"""分析以下对话，提取值得长期记忆的信息。

只提取用户明确表达的信息，不要猜测。
如果没有值得记忆的信息，只回复"无"。

对话：
{conversation_text}

请按以下格式输出（没有就写"无"）：
事实:
1. [具体事实]

画像:
- [属性名]: [属性值]"""

    try:
        response = llm.invoke([HumanMessage(content=extract_prompt)])
        content = response.content

        if "无" in content and len(content) < 10:
            return []

        new_memories = []

        # 提取事实
        for match in re.findall(r'\d+\.\s*(.+)', content):
            fact = match.strip()
            if fact and fact != "无":
                memory.add_fact(fact, source="对话提取")
                new_memories.append(fact)

        # 提取画像
        for key, value in re.findall(r'-\s*([^:：]+)[：:]\s*(.+)', content):
            memory.update_profile(key.strip(), value.strip())
            new_memories.append(f"画像: {key.strip()} = {value.strip()}")

        return new_memories
    except Exception as e:
        print(f"[记忆提取失败] {e}")
        return []


def _summarize_conversation(messages: List[BaseMessage], memory: AgentMemory) -> str:
    """总结对话，存入情景记忆"""
    conversation_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}"
        for m in messages
        if not isinstance(m, SystemMessage)
    )

    try:
        response = llm.invoke([HumanMessage(
            content=f"请用一句话总结以下对话的核心内容（30字以内）：\n\n{conversation_text}\n\n总结："
        )])
        summary = response.content.strip()
        memory.add_episode(summary, details=conversation_text[:500])
        return summary
    except Exception as e:
        print(f"[总结失败] {e}")
        return ""


def run(user_input: str, user_id: str = "default"):
    """
    带记忆的 Agent 运行循环。
    与 agent.py 的 run() 相比，多了 3 步：
      前: 检索记忆 → 注入 prompt
      中: 正常 ReAct 循环（复用 _parse_action / _execute_tool）
      后: 提取记忆 → 总结对话
    """
    # === 对话前：加载并检索记忆 ===
    memory = get_memory(user_id)
    relevant_memories = memory.retrieve_relevant(user_input)

    if relevant_memories:
        print(f"\n[记忆] 发现相关记忆：")
        print(relevant_memories)
        print()

    # 将记忆注入 system prompt（类比 JS 模板字符串拼接）
    system_content = SYSTEM_PROMPT
    if relevant_memories:
        system_content += f"\n\n## 相关记忆（请参考这些信息个性化回答）\n{relevant_memories}"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ]

    tool_called = False

    # === 对话中：ReAct 循环 ===
    for i in range(MAX_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"第 {i + 1} 轮")
        print("=" * 50)

        # 流式输出
        reply = ""
        for chunk in llm.stream(messages, stop=["Observation:"]):
            print(chunk.content, end="", flush=True)
            reply += chunk.content
        print()

        response = AIMessage(content=reply)
        action_type, action_content = _parse_action(reply)

        if action_type == "finish":
            if not tool_called:
                messages.append(response)
                messages.append(HumanMessage(
                    content="你还没有调用任何工具。请先使用工具查询信息，再继续。"
                ))
                continue

            print(f"\n{'='*50}")
            print("最终答案")
            print("=" * 50)
            answer = action_content.replace("\\n", "\n")
            print(answer)

            # === 对话后：提取并保存记忆 ===
            messages.append(response)
            print(f"\n[记忆] 保存本次对话...")

            new_memories = _extract_memories(messages, memory)
            if new_memories:
                print(f"  提取到 {len(new_memories)} 条新记忆:")
                for m in new_memories:
                    print(f"    - {m}")

            summary = _summarize_conversation(messages, memory)
            if summary:
                print(f"  对话摘要: {summary}")

            return answer

        if action_type == "tool":
            print(f"\n-> 执行工具: {action_content}")
            result = _execute_tool(action_content)
            print(f"<- 工具返回: {result[:200]}...")
            tool_called = True
            messages.append(response)
            messages.append(HumanMessage(content=f"Observation: {result}"))
        else:
            messages.append(response)
            messages.append(HumanMessage(content="请严格按照格式输出 Thought 和 Action。"))

    print("达到最大循环次数，Agent 停止。")
    _summarize_conversation(messages, memory)
    return None


def chat_loop():
    """
    交互式多轮对话。
    类比 JS 的 readline + while(true) 循环。
    """
    print("带记忆的 Agent 已启动")
    print("命令: /memory 查看记忆, /forget 清除记忆, /exit 退出")
    print("-" * 50)

    user_id = input("请输入用户ID (默认 default): ").strip() or "default"
    memory = get_memory(user_id)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue

            if user_input == "/exit":
                print("再见!")
                break

            if user_input == "/memory":
                profile = memory.get_profile()
                print(f"\n[记忆系统] 用户 '{user_id}'")
                print(f"  - 用户画像: {len(profile)} 项")
                for k, v in profile.items():
                    print(f"      {k}: {v}")
                print(f"  - 事实记忆: {len(memory.facts)} 条")
                for f in memory.facts:
                    print(f"      {f.content}")
                print(f"  - 情景记忆: {len(memory.episodes)} 条")
                for e in memory.episodes:
                    print(f"      {e.timestamp[:10]}: {e.content}")
                continue

            if user_input == "/forget":
                confirm = input("确定要清除所有记忆吗? (yes/no): ")
                if confirm.lower() == "yes":
                    import shutil
                    shutil.rmtree(memory.memory_dir, ignore_errors=True)
                    from memory import _memory_cache
                    _memory_cache.pop(user_id, None)
                    memory = get_memory(user_id)
                    print("记忆已清除")
                continue

            run(user_input, user_id=user_id)

        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    chat_loop()
