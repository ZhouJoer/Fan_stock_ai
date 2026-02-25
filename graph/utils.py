from langgraph.prebuilt import create_react_agent

def create_agent(llm, tools: list, system_prompt: str):
    # 根据版本使用 'prompt' 或 'messages_modifier' 或 'state_modifier'。
    # 根据检查，'prompt' 似乎是可用的参数。
    # 同时直接调用 create_react_agent 确保无递归。
    return create_react_agent(llm, tools, prompt=system_prompt)
