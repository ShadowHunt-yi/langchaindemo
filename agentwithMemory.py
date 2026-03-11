"""
带记忆的 Agent 循环
核心改进：
1. 每次对话前检索相关记忆
2. 对话中提取新记忆
3. 对话结束后总结并存储
"""

import re
from pathlib import Path
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

# 模拟依赖（实际使用时替换为你的真实实现）
class MockLLM:
    def invoke(self, messages):
        return type('obj', (object,), {'content': '无'})
    def stream(self, messages, stop=None):
        yield type('obj', (object,), {'content': 'Action: Finish[已查看记忆]'})

class AgentMemory:
    def __init__(self, user_id):
        self.user_id = user_id
        self.profile = {}
        self.facts = []
        self.episodes = []
    
    def retrieve_relevant(self, query):
        return "近期对话：查看记忆"
    
    def add_fact(self, fact, source):
        self.facts.append(fact)
    
    def update_profile(self, key, value):
        self.profile[key] = value
    
    def add_episode(self, summary, details):
        self.episodes.append(summary)
    
    def get_profile(self):
        return self.profile
    
    def get_recent_episodes(self, n):
        return self.episodes[-n:]

# 模拟全局变量
llm = MockLLM()
_memory_cache = {}

def get_memory(user_id):
    if user_id not in _memory_cache:
        _memory_cache[user_id] = AgentMemory(user_id)
    return _memory_cache[user_id]

# 读取增强版 prompt
PROMPT_FILE = Path(__file__).parent / "prompts" / "travel_agent_with_memory.txt"
# 兼容不存在的 prompt 文件
if PROMPT_FILE.exists():
    SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8")
else:
    SYSTEM_PROMPT = """你是一个旅游助手，需要先调用工具再回答问题。
可用工具：get_weather, get_attraction"""

MAX_ITERATIONS = 10

def _parse_action(text: str):
    """解析 LLM 的 Action（与之前相同）"""
    tool_match = re.search(r"Action:\s*(\w+\([^)]*\))", text, re.DOTALL)
    if tool_match:
        return "tool", tool_match.group(1).strip()

    finish_match = re.search(r"Action:\s*Finish\[(.+)\]", text, re.DOTALL)
    if finish_match:
        return "finish", finish_match.group(1).strip()

    return None, None

def _execute_tool(action_str: str) -> str:
    """执行工具（与之前相同）"""
    match = re.match(r"(\w+)\((.+)\)", action_str)
    if not match:
        return f"错误:无法解析工具调用 '{action_str}'"

    func_name = match.group(1)
    args_str = match.group(2)
    
    # 模拟工具映射
    TOOLS = {
        "get_weather": lambda city: f"{city}天气：晴",
        "get_attraction": lambda city, weather: f"{city} {weather} 景点推荐"
    }
    func = TOOLS.get(func_name)
    
    if not func:
        return f"错误:未知工具 '{func_name}'"

    kwargs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_str))
    
    try:
        return func(** kwargs)
    except TypeError as e:
        return f"错误:调用 {func_name} 参数不匹配 - {e}"

def extract_memories_from_conversation(
    messages: List[BaseMessage], 
    memory: AgentMemory
) -> List[str]:
    """使用 LLM 从对话中提取值得记忆的信息"""
    conversation_text = "\n".join([
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages[-6:]  # 只看最近几轮
    ])
    
    extract_prompt = f"""分析以下对话，提取任何值得长期记忆的信息。
这些信息可能包括：
- 用户的明确偏好（喜欢/不喜欢）
- 用户分享的个人事实（职业、家庭、习惯等）
- 重要的上下文信息（如用户纠正了AI的错误）

只提取具体、明确的信息，不要猜测。
如果没有任何值得记忆的信息，返回 "无"。

对话：
{conversation_text}

请按以下格式输出：
事实记忆:
1. [具体事实描述]
2. ...

用户画像更新:
- [属性名]: [属性值] (如: 偏好城市: 成都)

如果没有，直接写 "无"。"""

    try:
        response = llm.invoke([HumanMessage(content=extract_prompt)])
        content = response.content
        
        new_memories = []
        
        # 解析事实记忆
        fact_matches = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|$)', content, re.DOTALL)
        for fact in fact_matches:
            fact = fact.strip()
            if fact and fact != "无":
                memory.add_fact(fact, source="conversation_extraction")
                new_memories.append(fact)
        
        # 解析用户画像更新
        profile_matches = re.findall(r'-\s*([^:]+):\s*(.+)', content)
        for key, value in profile_matches:
            memory.update_profile(key.strip(), value.strip())
            new_memories.append(f"画像: {key} = {value}")
        
        return new_memories
        
    except Exception as e:
        print(f"[记忆提取失败] {e}")
        return []

def summarize_conversation(
    messages: List[BaseMessage],
    memory: AgentMemory
) -> str:
    """总结当前对话，存入情景记忆"""
    conversation_text = "\n".join([
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}..."
        for m in messages
    ])
    
    summary_prompt = f"""请用一句话总结以下对话的核心内容（30字以内）：

{conversation_text}

总结："""

    try:
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()
        memory.add_episode(summary, details=conversation_text[:500])
        return summary
    except Exception as e:
        print(f"[总结失败] {e}")
        return ""

def run(user_input: str, user_id: str = "default", conversation_id: Optional[str] = None):
    """
    带记忆的 Agent 运行循环
    
    Args:
        user_input: 用户输入
        user_id: 用户标识（用于区分不同用户的记忆）
        conversation_id: 对话会话ID（用于多轮对话保持上下文）
    """
    # 新增：提前拦截特殊命令
    if user_input.strip() == "/memory":
        memory = get_memory(user_id)
        print(f"\n[记忆系统] 已加载用户 '{user_id}' 的记忆")
        print(f"  - 用户画像: {len(memory.get_profile())} 项")
        print(f"  - 事实记忆: {len(memory.facts)} 条")
        print(f"  - 情景记忆: {len(memory.episodes)} 条")
        print(f"  - 对话摘要: 0 条")
        return "已查看记忆"
    
    # 1. 加载记忆
    memory = get_memory(user_id)
    
    # 2. 检索相关记忆
    relevant_memories = memory.retrieve_relevant(user_input)
    if relevant_memories:
        print(f"\n{'='*50}")
        print("[记忆检索] 发现相关记忆：")
        print(relevant_memories)
        print("=" * 50)
    
    # 3. 构建增强的系统 Prompt
    enhanced_system = SYSTEM_PROMPT
    if relevant_memories:
        enhanced_system += f"\n\n## 相关记忆（请利用这些信息更好地回答）\n{relevant_memories}"
    
    # 4. 初始化消息历史
    messages = [
        SystemMessage(content=enhanced_system),
        HumanMessage(content=user_input),
    ]
    
    tool_called = False
    full_reply = ""  # 记录完整回复用于后续记忆提取
    
    # 5. ReAct 循环
    for i in range(MAX_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"第 {i + 1} 轮")
        print("=" * 50)

        reply = ""
        for chunk in llm.stream(messages, stop=["Observation:"]):
            print(chunk.content, end="", flush=True)
            reply += chunk.content
        
        full_reply += reply + "\n"
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

            # 输出最终答案
            print(f"\n{'='*50}")
            print("最终答案")
            print("=" * 50)
            answer = action_content.replace("\\n", "\n")
            print(answer)
            
            # 6. 对话结束后：提取记忆并总结
            print(f"\n{'='*50}")
            print("[记忆处理] 保存本次对话...")
            
            # 添加最终答案到消息历史用于提取
            messages.append(response)
            
            # 提取新记忆
            new_memories = extract_memories_from_conversation(messages, memory)
            if new_memories:
                print(f"  提取到 {len(new_memories)} 条新记忆:")
                for m in new_memories:
                    print(f"    - {m}")
            
            # 总结对话
            summary = summarize_conversation(messages, memory)
            print(f"  对话摘要: {summary}")
            
            return answer

        if action_type == "tool":
            print(f"\n→ 执行工具: {action_content}")
            result = _execute_tool(action_content)
            print(f"← 工具返回: {result[:200]}...")
            tool_called = True

            messages.append(response)
            messages.append(HumanMessage(content=f"Observation: {result}"))
        else:
            messages.append(response)
            messages.append(HumanMessage(content="请严格按照格式输出 Thought 和 Action。"))

    print("达到最大循环次数，Agent 停止。")
    
    # 即使超时也要尝试保存记忆
    summarize_conversation(messages, memory)
    return None

def chat_loop():
    """交互式对话循环，支持多轮对话"""
    print("🤖 带记忆的 Agent 已启动")
    print("命令: /memory 查看记忆, /forget 清除记忆, /exit 退出")
    print("-" * 50)
    
    user_id = input("请输入用户ID (默认 default): ").strip() or "default"
    memory = get_memory(user_id)
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if not user_input:
                continue
                
            if user_input == "/exit":
                print("👋 再见！")
                break
            
            # 修复核心：处理 /memory 后直接 continue，不执行后续的 run()
            if user_input == "/memory":
                print(f"\n{'='*50}")
                print(f"[记忆系统] 已加载用户 '{user_id}' 的记忆")
                print(f"  - 用户画像: {len(memory.get_profile())} 项")
                print(f"  - 事实记忆: {len(memory.facts)} 条")
                print(f"  - 情景记忆: {len(memory.episodes)} 条")
                print(f"  - 对话摘要: 0 条")
                print("="*50)
                continue  # 关键：跳过后续的 run() 调用
            
            if user_input == "/forget":
                confirm = input("确定要清除所有记忆吗? (yes/no): ")
                if confirm.lower() == "yes":
                    import shutil
                    # 兼容不存在的目录
                    try:
                        shutil.rmtree(memory.memory_dir)
                    except:
                        pass
                    _memory_cache.pop(user_id, None)
                    print("✅ 记忆已清除")
                continue
            
            # 运行 Agent（仅处理非特殊命令）
            run(user_input, user_id=user_id)
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    chat_loop()