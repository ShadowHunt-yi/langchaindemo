"""
Agent 记忆系统 - 类似人类的短期记忆和长期记忆
"""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    content: str           # 记忆内容
    type: str             # 类型: "fact", "preference", "event", "summary"
    source: str           # 来源: 哪轮对话产生的
    timestamp: str        # 创建时间
    importance: float = 1.0  # 重要性分数 (0-1)
    access_count: int = 0    # 被检索次数（用于记忆强化）
    last_access: Optional[str] = None  # 最后访问时间
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


class AgentMemory:
    """
    Agent 的长期记忆系统
    存储位置: ~/.agent_memory/{user_id}/
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.memory_dir = Path.home() / ".agent_memory" / user_id
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 不同类型的记忆文件
        self.profile_file = self.memory_dir / "profile.json"
        self.facts_file = self.memory_dir / "facts.json"
        self.episodes_file = self.memory_dir / "episodes.json"
        self.summaries_file = self.memory_dir / "summaries.json"
        
        # 加载现有记忆
        self.profile: Dict[str, Any] = self._load_json(self.profile_file, {})
        self.facts: List[MemoryEntry] = self._load_memories(self.facts_file)
        self.episodes: List[MemoryEntry] = self._load_memories(self.episodes_file)
        self.summaries: List[MemoryEntry] = self._load_memories(self.summaries_file)
        
        print(f"[记忆系统] 已加载用户 '{user_id}' 的记忆")
        print(f"  - 用户画像: {len(self.profile)} 项")
        print(f"  - 事实记忆: {len(self.facts)} 条")
        print(f"  - 情景记忆: {len(self.episodes)} 条")
        print(f"  - 对话摘要: {len(self.summaries)} 条")
    
    def _load_json(self, path: Path, default: Any) -> Any:
        """加载 JSON 文件"""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default
    
    def _load_memories(self, path: Path) -> List[MemoryEntry]:
        """加载记忆条目列表"""
        data = self._load_json(path, [])
        return [MemoryEntry.from_dict(item) for item in data]
    
    def _save_json(self, path: Path, data: Any):
        """保存到 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_memories(self, path: Path, memories: List[MemoryEntry]):
        """保存记忆条目列表"""
        self._save_json(path, [m.to_dict() for m in memories])
    
    def _now(self) -> str:
        """当前时间字符串"""
        return datetime.now().isoformat()
    
    def _make_hash(self, content: str) -> str:
        """生成内容哈希，用于去重"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    # ============ 用户画像操作 ============
    
    def update_profile(self, key: str, value: Any, confidence: float = 1.0):
        """
        更新用户画像
        例如: update_profile("preferred_city", "成都")
              update_profile("dietary_restrictions", ["素食", "不吃辣"])
        """
        old_value = self.profile.get(key)
        self.profile[key] = {
            "value": value,
            "confidence": confidence,  # LLM 提取信息的置信度
            "updated_at": self._now(),
            "previous_value": old_value
        }
        self._save_json(self.profile_file, self.profile)
        print(f"[记忆] 更新画像: {key} = {value}")
    
    def get_profile(self, key: Optional[str] = None) -> Any:
        """获取用户画像"""
        if key:
            return self.profile.get(key, {}).get("value")
        return {k: v["value"] for k, v in self.profile.items()}
    
    # ============ 事实记忆操作 ============
    
    def add_fact(self, content: str, source: str = "", importance: float = 1.0):
        """
        添加事实记忆
        例如: "用户不喜欢下雨天出门", "用户是软件工程师"
        """
        # 去重检查
        content_hash = self._make_hash(content)
        for fact in self.facts:
            if self._make_hash(fact.content) == content_hash:
                # 已存在，更新重要性
                fact.importance = min(1.0, fact.importance + 0.1)
                fact.access_count += 1
                fact.last_access = self._now()
                self._save_memories(self.facts_file, self.facts)
                return
        
        entry = MemoryEntry(
            content=content,
            type="fact",
            source=source,
            timestamp=self._now(),
            importance=importance
        )
        self.facts.append(entry)
        self._save_memories(self.facts_file, self.facts)
        print(f"[记忆] 新增事实: {content[:50]}...")
    
    def search_facts(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """
        检索相关事实（简单关键词匹配，生产环境可用向量检索）
        """
        query_words = set(query.lower().split())
        scored = []
        
        for fact in self.facts:
            fact_words = set(fact.content.lower().split())
            overlap = len(query_words & fact_words)
            score = overlap * fact.importance  # 重叠词数 * 重要性
            
            if score > 0:
                scored.append((score, fact))
        
        # 按分数排序，返回 top_k
        scored.sort(reverse=True, key=lambda x: x[0])
        results = [fact for _, fact in scored[:top_k]]
        
        # 更新访问记录
        for fact in results:
            fact.access_count += 1
            fact.last_access = self._now()
        self._save_memories(self.facts_file, self.facts)
        
        return results
    
    # ============ 情景记忆（对话历史） ============
    
    def add_episode(self, summary: str, details: str = "", importance: float = 1.0):
        """添加情景记忆（对话摘要）"""
        entry = MemoryEntry(
            content=summary,
            type="event",
            source=details,
            timestamp=self._now(),
            importance=importance
        )
        self.episodes.append(entry)
        
        # 只保留最近 50 条情景记忆
        if len(self.episodes) > 50:
            self.episodes = sorted(
                self.episodes, 
                key=lambda x: (x.importance, x.timestamp), 
                reverse=True
            )[:50]
        
        self._save_memories(self.episodes_file, self.episodes)
        print(f"[记忆] 记录情景: {summary[:50]}...")
    
    def get_recent_episodes(self, n: int = 3) -> List[MemoryEntry]:
        """获取最近的情景记忆"""
        return sorted(self.episodes, key=lambda x: x.timestamp, reverse=True)[:n]
    
    # ============ 记忆检索接口 ============
    
    def retrieve_relevant(self, query: str, context: str = "") -> str:
        """
        检索与当前查询相关的记忆，返回格式化的记忆字符串
        这是给 LLM 使用的上下文增强
        """
        memories = []
        
        # 1. 检索相关事实
        facts = self.search_facts(query, top_k=3)
        if facts:
            memories.append("## 已知事实")
            for f in facts:
                memories.append(f"- {f.content}")
        
        # 2. 检索用户画像
        profile = self.get_profile()
        relevant_profile = {}
        for key, value in profile.items():
            # 简单判断相关性：查询词是否在 key 或 value 中
            if any(word in (key + str(value)).lower() for word in query.lower().split()):
                relevant_profile[key] = value
        
        if relevant_profile:
            memories.append("\n## 用户偏好")
            for k, v in relevant_profile.items():
                memories.append(f"- {k}: {v}")
        
        # 3. 最近的情景
        recent = self.get_recent_episodes(2)
        if recent:
            memories.append("\n## 近期对话")
            for e in recent:
                memories.append(f"- {e.timestamp[:10]}: {e.content}")
        
        return "\n".join(memories) if memories else ""
    
    def consolidate(self):
        """
        记忆巩固：定期执行，合并相似记忆，清理低价值记忆
        类似人类睡眠时的记忆整理
        """
        print("[记忆] 开始记忆巩固...")
        
        # 清理极低重要性的记忆
        threshold = 0.2
        old_facts = len(self.facts)
        self.facts = [f for f in self.facts if f.importance > threshold]
        if len(self.facts) < old_facts:
            print(f"  - 清理 {old_facts - len(self.facts)} 条低价值事实")
        
        # 保存
        self._save_memories(self.facts_file, self.facts)
        self._save_memories(self.episodes_file, self.episodes)


# 全局记忆缓存（避免重复加载）
_memory_cache: Dict[str, AgentMemory] = {}

def get_memory(user_id: str = "default") -> AgentMemory:
    """获取或创建记忆实例"""
    if user_id not in _memory_cache:
        _memory_cache[user_id] = AgentMemory(user_id)
    return _memory_cache[user_id]