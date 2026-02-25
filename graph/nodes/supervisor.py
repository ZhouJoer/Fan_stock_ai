import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from graph.state import AgentState
from llm import llm, tool_llm

members = ["Data_Researcher", "Trading_Strategy", "Financial_Analyst", "Decision_Maker", "Risk_Manager", "Quant_Trader", "Quant_Auditor", "ETF_Rotation_AI"]
system_prompt = (
    "你是一位严格的流程主管，负责监督股票分析团队的工作流。"
    "团队成员：{members}。"
    "\n\n"
    "【标准分析工作流】（传统方式）："
    "1. Data_Researcher (获取数据) -> 2. Trading_Strategy (技术分析) -> 3. Financial_Analyst (基本面/综合分析) -> 4. Decision_Maker (交易决策) -> 5. Risk_Manager (风控审核) -> FINISH"
    "\n\n"
    "【量化交易工作流】（当用户提到'量化'、'回测'、'策略'时使用）："
    "1. Data_Researcher (获取数据) -> 2. Quant_Trader (量化策略+回测) -> 3. Quant_Auditor (AI审计) -> FINISH"
    "\n\n"
    "【ETF AI轮动工作流】（当用户提到'ETF轮动'、'AI轮动'、'etf轮动'、'ai轮动'时使用）："
    "1. ETF_Rotation_AI (AI分析ETF轮动策略) -> FINISH"
    "\n\n"
    "为了决定下一步："
    "1. 检查对话历史中的【最后一条消息】内容和用户的原始请求。"
    "2. 如果用户请求包含'ETF轮动'、'AI轮动'、'etf轮动'、'ai轮动'、'ETF AI轮动'等关键词，直接路由给 'ETF_Rotation_AI'。"
    "3. 如果用户请求包含'量化'、'回测'、'策略测试'等关键词（且不包含ETF轮动），使用量化工作流。"
    "4. 如果最后一条消息是用户的初始请求（例如询问某股票），路由给 'Data_Researcher'。"
    "5. 【标准流程】如果最后消息是 'Data_Researcher' 且用户未提及量化，路由给 'Trading_Strategy'。"
    "6. 【量化流程】如果最后消息是 'Data_Researcher' 且用户提到量化，路由给 'Quant_Trader'。"
    "7. 如果最后一条消息是 'Trading_Strategy' 的输出，路由给 'Financial_Analyst'。"
    "8. 如果最后一条消息是 'Financial_Analyst' 的输出，路由给 'Decision_Maker'。"
    "9. 如果最后一条消息是 'Decision_Maker' 的输出，路由给 'Risk_Manager'。"
    "10. 如果最后一条消息是 'Risk_Manager' 的输出，路由给 'FINISH'。"
    "11. 如果最后一条消息是 'Quant_Trader' 的输出（量化回测报告），路由给 'Quant_Auditor'。"
    "12. 如果最后一条消息是 'Quant_Auditor' 的输出（AI审计报告），路由给 'FINISH'。"
    "13. 如果最后一条消息是 'ETF_Rotation_AI' 的输出（ETF AI轮动分析报告），路由给 'FINISH'。"
    "\n"
    "请仔细分析最后一条消息的内容特征和用户原始需求来判断当前处于流程的哪一步。"
)

options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "选择下一个角色。",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "鉴于上述对话，谁应该下一步行动？"
            "还是我们应该结束（FINISH）？请选择其中一个：{options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

def supervisor_node(state: AgentState):
    # 选项 1: 使用 bind_tools (但需要检查 LLM 提供商是否通过 OpenAI API 正确支持它)
    # 选项 2: 纯 Prompt 工程 (对通用模型最稳健)
    
    # 定义使用标准 OpenAI 格式的工具
    tool_def = {
        "type": "function",
        "function": function_def
    }
    
    try:
        # 尝试不同的绑定方法以兼容不同的 LangChain 版本
        supervisor_chain = None
        
        # 使用 tool_llm (deepseek-chat) 避免 reasoner 模型在 function calling 时的格式问题
        # DeepSeek R1 (reasoner) 不支持工具调用，或者不支持强制特定工具
        
        if hasattr(tool_llm, "bind_tools"):
             # 尝试 auto 而不是强制 route，兼容性更好
             supervisor_chain = prompt | tool_llm.bind_tools([function_def], tool_choice="auto")
        elif hasattr(tool_llm, "bind_functions"):
             supervisor_chain = prompt | tool_llm.bind_functions(functions=[function_def])
        else:
             # 回退：直接通过 kwargs 绑定
             supervisor_chain = prompt | tool_llm.bind(functions=[function_def])

        response = supervisor_chain.invoke(state)
        
        # 手动解析函数调用结果
        next_agent = None
        # if response.today_calls: # basic check  --> 'today_calls' was a typo in previous edit
        #      pass
             
        if response.tool_calls:
             # 如果使用 bind_tools，我们会得到 tool_calls
             args = response.tool_calls[0]['args']
             next_agent = args["next"]
        elif response.additional_kwargs.get("function_call"):
            args = json.loads(response.additional_kwargs["function_call"]["arguments"])
            next_agent = args["next"]
        else:
            # 回退：模型可能直接输出了 "FINISH" 或类似内容 (或者 CoT 的内容)
            content = response.content.strip()
            # 简单的清理，假设最后的内容是结果
            lines = content.split('\n')
            for line in reversed(lines):
                clean_line = line.strip().replace('"', '').replace("'", "")
                if clean_line in options:
                    next_agent = clean_line
                    break
            
            if not next_agent:
                 next_agent = "FINISH"
        
        # --- Debug Info ---
        print(f"\n[Supervisor Debug]")
        print(f"  Last Message Preview: {str(state['messages'][-1].content)[:100]}...")
        print(f"  Route Decision: {next_agent}")
        # ------------------

    except Exception as e:
        # 错误静默处理，或者只打印简短警告，以免干扰用户体验
        print(f"[Supervisor Error] Tool calling failed: {e}")
        
        # 如果绑定完全失败，进行显式回退
        # 尝试简单的文本 Prompt
        try:
             # 简单提取 - 注意这里也应该用 tool_llm 或者 llm
             # 如果上一步 llm 报错，这里换回 tool_llm
             raw_response = (prompt | tool_llm).invoke(state)
             content = raw_response.content.strip()
             # 简单启发式匹配
             for member in members:
                 if member in content:
                     next_agent = member
                     break
             else:
                 next_agent = "FINISH"
        except:
             next_agent = "FINISH"

    return {"next": next_agent}
