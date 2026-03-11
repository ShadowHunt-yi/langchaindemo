"""
入口文件 — 类比 JS 的 index.js
只负责接收输入、调用 agent，不放任何业务逻辑。

Python 知识点:
  if __name__ == "__main__" 等价于 JS 里判断"这个文件是不是被直接运行的"
  - 直接运行 python main.py → __name__ == "__main__" → 执行
  - 被别的文件 import → 不执行
"""
from agent import run


def main():
    user_input = input("请输入你的问题: ")
    run(user_input)


if __name__ == "__main__":
    main()
