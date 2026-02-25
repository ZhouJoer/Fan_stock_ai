"""命令行入口：python -m graph.agents.news_agent "你的问题""" 

from __future__ import annotations

import sys

from .core import run_news_agent


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m graph.agents.news_agent \"查询内容\"")
        raise SystemExit(2)

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("查询内容不能为空")
        raise SystemExit(2)

    print(run_news_agent(query))


if __name__ == "__main__":
    main()
