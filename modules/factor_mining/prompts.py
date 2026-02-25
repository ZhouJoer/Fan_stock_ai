"""
因子挖掘工作流：LLM 选因子与策略/轮仓报告用 prompt 与 schema；
并定义各 node 分工与 AI Agent 编排说明，供编排器/Agent 使用。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Structured output 用 Pydantic（与 LangChain 兼容）
try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None  # type: ignore
    Field = None  # type: ignore


# ---------- 各 Node 分工（供 AI Agent 编排与说明） ----------
NODE_ROLES = """
## 因子挖掘工作流节点分工

| 节点名 | 职责 | 输入(State) | 输出(State) |
|--------|------|-------------|-------------|
| load_data | 加载股票行情与基准数据 | universe_codes, days, label_horizon, benchmark_code | data_dict, benchmark_returns |
| compute_factors | 构建训练样本、计算因子及截面相关矩阵 | data_dict, days, label_horizon | factor_df, factor_corr_matrix, factor_names |
| try_combinations | （dual/multi 模式）枚举低相关因子组合，逐组训练+回测+alpha/beta，按年化 alpha + val_rank_ic 选最佳；选因子时遵循：低相关、可解释、非单调则钟形 | factor_df, factor_names, factor_corr_matrix, mode, benchmark_returns, label_horizon, rebalance_freq | selected_factors, train_result, strategy_returns, alpha_beta, backtest_stats |
| select_factors_llm | （single 模式）由 LLM 选 1 个因子及可选钟形变换；选因子时遵循：低相关、可解释、非单调则钟形 | factor_names, factor_corr_matrix, mode | selected_factors, bell_transforms |
| apply_transforms | 对选中因子做钟形变换，得到最终用于训练的列名 | factor_df, selected_factors, bell_transforms | factor_df(更新), selected_factors(含 _bell) |
| train | 用选中因子训练线性 Softmax 权重 | factor_df, selected_factors | train_result |
| alpha_beta | 按调仓周期回测（选 top-N 等权）、与 horizon 对齐的基准做 CAPM | factor_df, train_result, benchmark_returns, label_horizon, rebalance_freq | strategy_returns, alpha_beta, backtest_stats |
| report | 根据结果生成策略逻辑与轮仓逻辑两段文字 | train_result, selected_factors, alpha, beta, alpha_beta, metrics | strategy_logic, rotation_logic |

**分支规则**：
- 从 compute_factors 出来：mode 为 dual 或 multi → 进入 try_combinations，否则进入 select_factors_llm。
- try_combinations 结束后直接进入 report。
- select_factors_llm 后：若有 bell_transforms → apply_transforms → train，否则直接 train；train → alpha_beta → report。
"""


def get_agent_orchestration_prompt() -> str:
    """返回给 AI Agent 的编排说明：如何驱动因子挖掘工作流、各节点顺序与分支。"""
    return """你是因子挖掘工作流的编排 Agent。请根据「节点分工」理解各步骤职责，并按以下流程驱动或向用户说明：

**1. 入口与必选输入**
- 用户需提供：股票池代码列表 universe_codes；可选：days（样本天数）、label_horizon（收益周期）、rebalance_freq（调仓周期，1=日频，5=周频）、mode（single/dual/multi）、benchmark_code（基准代码）。
- 工作流从 load_data 开始，依次执行，直至 report 结束。

**2. 执行顺序与分支**
- load_data → compute_factors（必执行）。
- compute_factors 之后：
  - 若 mode 为 dual 或 multi：执行 try_combinations（枚举多组因子组合并选最佳），然后跳到 report。
  - 若 mode 为 single：执行 select_factors_llm；若返回的 bell_transforms 非空则执行 apply_transforms，再执行 train → alpha_beta → report；否则直接 train → alpha_beta → report。

**3. 调仓与回测**
- 回测在 alpha_beta 节点（以及 try_combinations 内部）按 rebalance_freq 调仓：仅在调仓日选 top-N 等权，持有至下一调仓日，收益与基准均按 label_horizon 对齐后算 alpha/beta。
- 若用户未指定 rebalance_freq，默认 1（每日调仓）；建议与 label_horizon 一致或为约数（如 horizon=5 则 rebalance_freq=5）。

**4. 你的职责**
- 根据用户意图决定或建议 mode、rebalance_freq 等参数；编排或推荐时优先考虑能产生低相关因子组合的 mode 与候选集（如 multi 模式下强调跨类别选因子）。
- 按上述顺序触发或说明各节点；若某节点返回 error，向用户说明并建议修正输入或跳过该分支。
- 在 report 之后，可汇总：选中因子、权重、alpha/beta、回测统计、策略逻辑与轮仓逻辑。
- 选因子与推荐组合时遵循：① 各因子间相关性尽量低（|r| 小、跨类别）；② 尽量选用可解释性强的因子；③ 若因子与收益非单调，则采用钟形变换。
""" + NODE_ROLES


def get_orchestration_tasks_prompt(context: dict) -> str:
    """返回让 AI Agent 生成「深度编排」任务列表的 prompt。context 可含：universe_size, days, benchmark_code, user_preference（可选）。"""
    universe_size = context.get("universe_size", 0)
    days = context.get("days", 252)
    benchmark_code = context.get("benchmark_code", "510300")
    user_preference = context.get("user_preference") or "深度挖掘多组因子组合，覆盖不同模式与调仓频率"
    return f"""你是因子挖掘的**编排 Agent**。请根据当前上下文，生成一批**挖掘任务**，供系统依次执行并汇总最佳结果。

**当前上下文**：
- 股票池规模：约 {universe_size} 只
- 样本天数：{days} 天
- 基准代码：{benchmark_code}
- 用户意图：{user_preference}

**任务参数说明**：
- mode：必选其一。`single`=单因子；`dual`=双因子（低相关）；`multi`=多因子（3～5 个，低相关、可解释、非单调可钟形）。
- label_horizon：预测收益周期（交易日数），常用 1、3、5。
- rebalance_freq：调仓周期（交易日数），1=每日调仓，5=约周频；建议与 label_horizon 一致或为约数。

**要求**：
1. 生成 5～12 个任务，覆盖多种 mode（以 dual 与 multi 为主，便于 try_combinations 枚举组合）、label_horizon（如 1、3、5）、rebalance_freq（如 1、5）。
2. 任务之间要有差异，避免重复；可组合例如：multi+horizon=5+rebal=1、multi+horizon=5+rebal=5、dual+horizon=3+rebal=5 等。
3. 只输出任务列表的 JSON，不要其他解释。每个任务包含 mode（字符串）、label_horizon（整数）、rebalance_freq（整数）。
""" + "\n输出 JSON 格式：{\"tasks\": [{\"mode\": \"multi\", \"label_horizon\": 5, \"rebalance_freq\": 1}, ...]}\n"


def get_agent_next_step_prompt(state_summary: dict) -> str:
    """Agent 逐步决策：下一步应挖掘/选择什么因子，或计算权重/结束。从低相关、可解释出发；分层收益判断是否钟形。"""
    factor_names = state_summary.get("factor_names") or []
    corr_text = state_summary.get("corr_matrix_text") or ""
    main_factor = state_summary.get("main_factor")
    other_factors = state_summary.get("other_factors") or []
    bell_transforms = state_summary.get("bell_transforms") or []
    quantile_returns = state_summary.get("quantile_returns_per_factor") or {}
    suggest_bell = state_summary.get("suggest_bell_per_factor") or {}
    quality_per_factor = state_summary.get("quality_per_factor") or {}
    factor_mode = state_summary.get("factor_mode", "multi")
    max_factors = state_summary.get("max_factors", 5)
    remaining_slots = state_summary.get("remaining_slots", max_factors)
    tried_combos = state_summary.get("tried_combos") or []
    trial_idx = state_summary.get("trial_idx", 0)
    forced_main = state_summary.get("forced_main_factor")
    step = state_summary.get("step_count", 0)
    max_steps = state_summary.get("max_steps", 15)
    last_metrics = state_summary.get("last_metrics")

    selected = ([main_factor] if main_factor else []) + list(other_factors)
    selected = [f for f in selected if f]
    mode_label = {"single": "单因子（最多 1 个）", "dual": "双因子（最多 2 个）", "multi": "多因子（最多 5 个）"}.get(factor_mode, f"最多 {max_factors} 个")

    forced_main_note = f"⚙️ 本次探索主因子已由系统固定为 **{forced_main}**，请勿更改主因子，只需选择最优**辅助因子**。" if forced_main else ""
    lines = [
        "你是因子挖掘的**逐步编排 Agent**。请根据当前状态决定**下一步**动作。",
        "核心原则：① 低相关性优先；② 可解释性优先（IC |均值| 越大越有效）；③ spread≈0 或 IC≈0 的因子**禁止加入**；④ 非单调（中间高两端低）才做钟形变换。",
        "",
        f"**挖掘模式：{mode_label}，当前第 {trial_idx + 1} 次探索，已选 {len(selected)} 个，剩余名额 {remaining_slots} 个。**",
        forced_main_note,
        "⚠️ 严格限制：达到模式上限后不得再 add_factor，直接 compute_weights 或 finish。",
        "",
        "**当前状态**：",
        f"- 已选主因子：{main_factor or '（未设）'}",
        f"- 已选其它因子：{', '.join(other_factors) or '无'}",
        f"- 已做钟形变换：{', '.join(bell_transforms) or '无'}",
        f"- 步数：{step}/{max_steps}",
    ]
    if corr_text:
        lines.append("")
        lines.append("**因子间相关系数矩阵**（|r| 越小越不相关，同族因子只选一个）：")
        lines.append(corr_text)
    if factor_names:
        lines.append("")
        lines.append("**候选因子质量一览**（spread=分层收益极差；IC=截面预测能力均值；IC_IR=IC稳定性；越大越好）：")
        for f in factor_names:
            qr = quantile_returns.get(f) or {}
            sb = suggest_bell.get(f, False)
            qi = quality_per_factor.get(f, {})
            spread = qi.get("spread", 0.0)
            direction = qi.get("direction", "unknown")
            ic = qi.get("ic", 0.0)
            ic_ir = qi.get("ic_ir", 0.0)
            in_sel = "✓已选" if f in selected else ""
            qr_str = ", ".join(f"{k}:{round(v,5)}" for k, v in sorted(qr.items())) if qr else "—"
            bell_hint = "→建议钟形" if sb else ""
            effective = "" if (spread > 0 and abs(ic) >= 0.01) else " ⚠️无效"
            lines.append(f"  - {f}{' ' + in_sel if in_sel else ''}{effective}: spread={spread:.5f}, IC={ic:.5f}, IC_IR={ic_ir:.3f}, dir={direction}{bell_hint}")
    if tried_combos:
        lines.append("")
        lines.append("**已尝试过的组合（请避免重复选择相同因子集合）**：")
        for tc in tried_combos[-5:]:  # 最多展示最近 5 组
            lines.append(f"  - {tc}")
    if last_metrics:
        lines.append("")
        lines.append("**上一轮 compute_weights 后的验证集指标（val_ic / sharpe 等）**：")
        lines.append(str(last_metrics))
        val_ic = last_metrics.get("val_ic", 0.0) if isinstance(last_metrics, dict) else 0.0
        if isinstance(val_ic, (int, float)) and val_ic < 0.005:
            lines.append("⚠️ val_ic 接近 0，当前组合预测能力弱；若已达上限请直接 finish，否则考虑替换。")
    lines.extend([
        "",
        "**决策规则**：",
        f"1. 若无主因子 → set_main，选 spread 最大且可解释性强的因子。",
        f"2. 若有主因子且剩余名额>0 → add_factor，选与已选因子相关性低（|r|<0.5）、spread 排名靠前、跨类别的因子；若该因子 direction=bell 则 use_bell=true。",
        f"3. spread=0 的因子**禁止**加入（无区分度）。",
        f"4. 若已选因子≥1 且尚未 compute_weights → compute_weights。",
        f"5. compute_weights 后若 val_ic>0.01 且剩余名额>0 → 可继续 add_factor；否则 finish。",
        f"6. 剩余名额=0 → 直接 finish（不要再 add_factor）。",
        "",
        "**输出格式**：仅输出一个 JSON 对象，不要任何说明或 markdown。",
        "示例：{\"action\": \"set_main\", \"factor_name\": \"momentum_20\", \"reason\": \"spread 最大，可解释\"}",
    ])
    return "\n".join(lines)


def get_factor_selection_prompt(
    mode: str,
    factor_names: List[str],
    corr_matrix_text: str,
    suggest_bell: List[str],
) -> str:
    """选因子时的系统 prompt：要求按 mode 选 1/2/多 个因子，且彼此相关性尽量低。"""
    mode_desc = {
        "single": "1 个因子",
        "dual": "2 个因子，且两因子相关性必须低（|r|<0.6），禁止选同族多窗口因子",
        "multi": "3～5 个因子，整体彼此相关性尽量低，禁止选同族多窗口因子",
    }.get(mode, "1 个因子")
    return f"""你是一个量化因子选型助手。请根据以下信息选出用于做多因子选股的因子组合。

**选取模式**：{mode_desc}

**候选因子列表**（共 {len(factor_names)} 个）：
{', '.join(factor_names)}

**因子间相关系数矩阵**（截面相关，数值越接近 0 越不相关）：
{corr_matrix_text}

**重要约束**：
- 任意两两因子相关系数 |r| 必须 < 0.6，否则视为高相关、不可同时入选。优先选择相关系数矩阵中两两 |r| 更小的因子组合；组合内因子应尽量来自不同类别（动量、波动、量价、估值等），避免同族多窗口同时入选。
- 禁止同时选择“同族、多窗口”的因子，例如：momentum_20 与 momentum_60 高度相关，只能二选一；同理 turnover_proxy 与 turnover_ratio、多周期同类因子等只能择一。应跨类别组合（如动量 + 波动 + 量价 + 估值），以降低冗余、提高区分度。
- 优先选择业务可解释性强的因子（如动量、波动率、PE、换手、量价、RSI 等有明确金融含义的指标），便于策略说明与风控；避免纯统计或难以解释的衍生项。

**钟形变换规则**：若因子与收益的关系非单调（并非越高/越低越好，而是中间段更优），必须在 bell_transforms 中列出该因子。钟形变换后得分在中间极值处最优，适合反转或适度暴露类因子（如 RSI、振幅等）。以下因子通常适合钟形变换，若入选则默认加入 bell_transforms：{', '.join(suggest_bell)}。

请严格按模式数量选取因子，且满足上述相关性约束与可解释性偏好。对建议列表中的因子若入选则默认加入 bell_transforms，除非有充分理由认为其与收益单调。
输出 JSON：{{"selected_factors": ["因子名1", "因子名2", ...], "bell_transforms": ["因子名"]}}"""


def get_report_generation_prompt(
    selected_factors: List[str],
    weights: dict,
    alpha: float,
    beta: float,
    metrics: dict,
    rebalance_freq: int = 1,
) -> str:
    """生成策略逻辑与轮仓逻辑的 prompt。"""
    rebalance_desc = "每日调仓" if rebalance_freq <= 1 else f"每 {rebalance_freq} 个交易日调仓（约{'周' if rebalance_freq >= 5 else '双周'}频）"
    return f"""根据以下因子挖掘结果，生成两段简短中文说明。

**选中因子及权重**：{selected_factors}
权重：{weights}

**CAPM 结果**：alpha={alpha}, beta={beta}

**训练指标**：{metrics}

**当前回测调仓设定**：{rebalance_desc}。请在轮仓逻辑中明确写出调仓频率与再平衡方式。

请输出两段文字（不要用 markdown 标题）：
1. 策略逻辑：选股依据、因子含义与权重逻辑、预期风格（价值/动量/波动等），2～4 句。
2. 轮仓逻辑：调仓频率（日/周/月或每 N 个交易日）、排序与截断规则（如 TopN）、换手控制、等权再平衡方式，2～4 句。

直接输出两段，用换行分隔，不要其他前缀或编号。"""


if BaseModel is not None and Field is not None:

    class FactorSelectionOutput(BaseModel):
        """LLM 选因子时的结构化输出。"""
        selected_factors: List[str] = Field(description="选中的因子名称列表，数量需符合 single/dual/multi")
        bell_transforms: List[str] = Field(default_factory=list, description="建议做钟形变换的因子名")

    class OrchestrationTaskSpec(BaseModel):
        """编排单条任务参数。"""
        mode: str = Field(description="single | dual | multi")
        label_horizon: int = Field(description="预测收益周期（交易日数），如 1、3、5", ge=1, le=20)
        rebalance_freq: int = Field(description="调仓周期（交易日数），1=日频，5=周频", ge=1, le=20)

    class OrchestrationPlanOutput(BaseModel):
        """编排 Agent 输出的任务列表。"""
        tasks: List[OrchestrationTaskSpec] = Field(description="待执行的挖掘任务列表，每项含 mode、label_horizon、rebalance_freq")

    class AgentNextStepOutput(BaseModel):
        """逐步编排 Agent 的下一步动作。"""
        action: str = Field(description="set_main | add_factor | remove_factor | compute_weights | finish")
        factor_name: Optional[str] = Field(default=None, description="set_main/add_factor/remove_factor 时的因子名")
        use_bell: Optional[bool] = Field(default=None, description="add_factor 时是否对该因子做钟形变换，可参考分层收益")
        reason: Optional[str] = Field(default=None, description="简短理由")

else:
    FactorSelectionOutput = None  # type: ignore
    OrchestrationTaskSpec = None  # type: ignore
    OrchestrationPlanOutput = None  # type: ignore
    AgentNextStepOutput = None  # type: ignore


# ---------------------------------------------------------------------------
# Orchestration Agent — 编排 Agent：规划因子组合方案
# ---------------------------------------------------------------------------

def get_orchestration_agent_prompt(
    quality_per_factor: Dict[str, Any],
    high_corr_text: str,
    factor_mode: str,
    max_factors: int,
    n_trials: int,
) -> str:
    """
    编排 Agent prompt：根据因子质量，规划 n_trials 个因子组合供评价 Agent 测试。
    注意：钟形变换由评价 Agent 根据数据自动决定，编排 Agent 只管组合哪些因子。
    """
    mode_label = {"single": "单因子", "dual": "双因子", "multi": "多因子"}.get(factor_mode, f"{max_factors}因子")
    lines = [
        "你是量化因子挖掘的**编排 Agent（Orchestrator）**。",
        "你的职责是：综合考虑因子质量与相关性，规划多套因子组合方案，交给评价 Agent 逐一测试。",
        "注意：你**只负责规划组合**，是否做钟形变换由评价 Agent 根据实际分层收益数据自动决定，无需你指定。",
        "",
        f"**挖掘模式**：{mode_label}  |  **每组因子数**：{'3～5' if factor_mode == 'multi' else max_factors}  |  **需规划**：{n_trials} 个组合",
        "",
        "## 候选因子质量一览（按 |IC| 降序）",
        "格式：因子名 | spread（分层收益极差）| IC（截面预测力，|IC|>0.01有效）| IC_IR（IC稳定性）| dir（收益方向）| 有效",
    ]
    for fname, q in sorted(
        quality_per_factor.items(),
        key=lambda x: abs(float(x[1].get("ic", 0.0) or 0.0)),
        reverse=True,
    ):
        spread = float(q.get("spread", 0.0) or 0.0)
        ic = float(q.get("ic", 0.0) or 0.0)
        ic_ir = float(q.get("ic_ir", 0.0) or 0.0)
        direction = str(q.get("direction", "?"))
        eff = "✓" if (abs(ic) > 0.005 and spread > 0) else "✗"
        lines.append(f"  {fname}: spread={spread:.4f}, IC={ic:.4f}, IC_IR={ic_ir:.3f}, dir={direction}, 有效={eff}")

    lines.append("")
    if high_corr_text:
        lines += [
            "## 高相关因子对（|r| ≥ 0.4，同一组合内应避免同时使用）",
            high_corr_text,
            "",
        ]

    lines += [
        "## 规划要求",
        "1. 每个组合包含 **3～5** 个因子，使用上表中实际存在的因子名",
        "2. 不同组合使用**不同主因子**（第一个因子），保证方案多样性",
        "3. 同一组合内因子两两相关性低（参考高相关对列表，避免 |r| ≥ 0.5 的配对）",
        "4. 优先选 '有效=✓' 的因子（|IC|>0.005 且 spread>0）",
        "5. 覆盖不同 alpha 来源：趋势/动量、均值回归/反转、低波动/质量、价值/规模等",
        "6. 每组的主因子（第一个）应是该组中 spread 或 |IC| 最大的",
        "7. 排除 spread=0 且 IC=0 的因子（无区分度）",
        "",
        "**输出纯 JSON，不含任何说明文字。格式**：",
        '{"combinations": [{"factors": ["f1", "f2"], "reason": "简短理由"}, ...]}',
        f"请输出恰好 **{n_trials}** 个组合。",
    ]
    return "\n".join(lines)


def parse_orchestration_json(text: str) -> Optional[List[Dict[str, Any]]]:
    """从编排 Agent 输出中解析因子组合列表。返回 [{factors, reason}, ...] 或 None。"""
    if not text:
        return None
    text = text.strip()
    if "```" in text:
        for marker in ("```json", "```"):
            if marker in text:
                start = text.find(marker) + len(marker)
                end = text.find("```", start)
                text = text[start: end if end != -1 else len(text)].strip()
                break
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        combos = obj.get("combinations") or []
        result = []
        for c in combos:
            factors = [str(f).strip() for f in (c.get("factors") or []) if f]
            reason = str(c.get("reason") or "")
            if factors:
                result.append({"factors": factors, "reason": reason})
        return result if result else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Reviewer Agent — 审查挖掘结果，剥离自编排 Agent 的质量判断职能
# 两阶段调用：先选 trial（phase1），再仅用该 trial 的指标生成评审/策略/轮仓（phase2），
# 保证评审意见中的数字来源唯一，与挖掘报告一致。
# ---------------------------------------------------------------------------

def get_reviewer_prompt_phase1(
    trial_summaries: List[Dict[str, Any]],
    quality_per_factor: Dict[str, Any],
    corr_text: str,
    factor_mode: str,
    rebalance_freq: int,
    label_horizon: int,
) -> str:
    """Phase1：只让 LLM 选出最优 trial，并给出结论/质量/可靠性/市值范围。不生成 comments/strategy/rotation。"""
    mode_label = {"single": "单因子", "dual": "双因子", "multi": "多因子"}.get(factor_mode, "多因子")
    rebalance_desc = "每日" if rebalance_freq <= 1 else f"每 {rebalance_freq} 交易日"
    n_trials = len(trial_summaries)

    lines = [
        "你是量化策略**审查 Agent**（Reviewer）**第一步**：仅根据下方各 trial 数据，选出综合最优的一个 trial。",
        "",
        f"**挖掘模式**：{mode_label}  |  **标签周期**：{label_horizon} 交易日  |  **调仓频率**：{rebalance_desc}",
        "",
        "---",
        f"## 各次探索结果（共 {n_trials} 次）",
    ]

    for i, ts in enumerate(trial_summaries):
        combo = ts.get("combo") or []
        weights = ts.get("weights") or {}
        bs = ts.get("backtest_stats") or {}
        metrics = ts.get("metrics") or {}
        ab = ts.get("alpha_beta") or {}
        lines.append(f"\n### Trial {i}（0-indexed）")
        lines.append(f"- 因子组合：{combo}")
        lines.append(f"- 线性权重：{weights}")
        lines.append(f"- 年化夏普：{bs.get('sharpe_annual', '—')}  |  总收益：{round(float(bs.get('total_return', 0) or 0) * 100, 2)}%  |  最大回撤：{round(float(bs.get('max_drawdown', 0) or 0) * 100, 2)}%")
        lines.append(f"- Alpha(年化)：{ab.get('annualized_alpha', '—')}  |  Beta：{ab.get('beta', '—')}  |  R²：{ab.get('r_squared', '—')}")
        lines.append(f"- 验证集 IC：{metrics.get('val_ic', metrics.get('val_rank_ic', '—'))}  |  训练集 IC：{metrics.get('train_ic', metrics.get('train_rank_ic', '—'))}")
        for f in combo:
            base_f = f[:-5] if f.endswith("_bell") else f
            q = quality_per_factor.get(base_f) or quality_per_factor.get(f, {})
            if q:
                lines.append(f"  - {f}: spread={q.get('spread', '—')}, IC={q.get('ic', '—')}, IC_IR={q.get('ic_ir', '—')}, dir={q.get('direction', '—')}")

    if corr_text:
        lines.extend(["", "## 因子间相关系数矩阵", corr_text])

    lines.extend([
        "",
        "---",
        "## 本步要求（仅输出选择与结论，不写评审意见与策略说明）",
        "1. **选出最优 trial**：selected_trial_idx 为 0 到 " + str(n_trials - 1) + " 的整数。",
        "2. **质量评分**（0-10）、**可靠性**（高/中/低）、**综合结论**（推荐/谨慎推荐/不推荐）、**适用市值**（大盘/中小盘/全市场）。",
        "3. 结论须符合：年化 Alpha≤0 不得推荐；夏普<0.3 或 Alpha<-0.3 须不推荐；回撤<-20% 不得推荐。",
        "",
        "**输出格式**：仅输出一个 JSON，不含其他文字。",
        '{"selected_trial_idx": 0, "verdict": "推荐", "quality_score": 7.5, "reliability": "中", "cap_recommendation": "全市场"}',
    ])
    return "\n".join(line for line in lines)


def get_reviewer_prompt_phase2(
    selected_trial_summary: Dict[str, Any],
    quality_per_factor: Dict[str, Any],
    rebalance_freq: int,
    factor_mode: str,
    label_horizon: int,
) -> str:
    """Phase2：仅传入已选中的那一个 trial 的指标，让 LLM 生成策略说明、轮仓说明、评审意见、风险提示。数字来源唯一。"""
    mode_label = {"single": "单因子", "dual": "双因子", "multi": "多因子"}.get(factor_mode, "多因子")
    rebalance_desc = "每日" if rebalance_freq <= 1 else f"每 {rebalance_freq} 交易日"
    combo = selected_trial_summary.get("combo") or []
    weights = selected_trial_summary.get("weights") or {}
    bs = selected_trial_summary.get("backtest_stats") or {}
    metrics = selected_trial_summary.get("metrics") or {}
    ab = selected_trial_summary.get("alpha_beta") or {}

    lines = [
        "你是量化策略**审查 Agent**（Reviewer）**第二步**：下方是**唯一**一份本组合的数据。请仅根据下方表格生成策略说明、轮仓说明、评审意见与风险提示。",
        "**重要**：评审意见（comments）和风险提示（risks）中若引用数字，必须且只能来自下方「本组合指标」表格，不得编造或使用其他来源。",
        "",
        f"**挖掘模式**：{mode_label}  |  **调仓频率**：{rebalance_desc}",
        "",
        "---",
        "## 本组合指标（唯一数据来源）",
        f"- 因子组合：{combo}",
        f"- 线性权重：{weights}",
        f"- 年化夏普：{bs.get('sharpe_annual', '—')}  |  总收益：{round(float(bs.get('total_return', 0) or 0) * 100, 2)}%  |  最大回撤：{round(float(bs.get('max_drawdown', 0) or 0) * 100, 2)}%",
        f"- Alpha(年化)：{ab.get('annualized_alpha', '—')}  |  Beta：{ab.get('beta', '—')}  |  R²：{ab.get('r_squared', '—')}",
        f"- 验证集 IC：{metrics.get('val_ic', metrics.get('val_rank_ic', '—'))}  |  训练集 IC：{metrics.get('train_ic', metrics.get('train_rank_ic', '—'))}",
    ]
    for f in combo:
        base_f = f[:-5] if f.endswith("_bell") else f
        q = quality_per_factor.get(base_f) or quality_per_factor.get(f, {})
        if q:
            lines.append(f"  - {f}: spread={q.get('spread', '—')}, IC={q.get('ic', '—')}, IC_IR={q.get('ic_ir', '—')}, dir={q.get('direction', '—')}")

    lines.extend([
        "",
        "---",
        "## 要求",
        "1. **策略逻辑**（2-4 句）：选股依据、因子含义与预期风格。",
        "2. **轮仓逻辑**（2-3 句）：调仓频率、排序截断、等权再平衡。",
        "3. **评审意见**（3-5 条）：优点与不足。可引用具体数字，但**必须全部来自上方「本组合指标」**（如年化夏普、总收益、最大回撤、Alpha、Beta、验证集 IC 等）。",
        "4. **风险提示**（1-3 条）：可引用上方数字。",
        "",
        "**输出格式**：仅输出一个 JSON，不含其他文字。",
        '{"strategy_logic": "策略说明...", "rotation_logic": "轮仓说明...", "comments": ["优点或不足，可带数字但须来自上表"], "risks": ["风险提示"]}',
    ])
    return "\n".join(line for line in lines)


def get_reviewer_prompt(
    trial_summaries: List[Dict[str, Any]],
    quality_per_factor: Dict[str, Any],
    corr_text: str,
    factor_mode: str,
    rebalance_freq: int,
    label_horizon: int,
) -> str:
    """
    单阶段审查 prompt（仅用于两阶段失败时的回退）。正常流程用 phase1 + phase2。
    """
    mode_label = {"single": "单因子", "dual": "双因子", "multi": "多因子"}.get(factor_mode, "多因子")
    rebalance_desc = "每日" if rebalance_freq <= 1 else f"每 {rebalance_freq} 交易日"
    n_trials = len(trial_summaries)
    lines = [
        "你是量化策略**审查 Agent**（Reviewer）。根据下方各 trial 选出最优一个，并输出结论与评审。",
        "",
        f"**挖掘模式**：{mode_label}  |  **调仓频率**：{rebalance_desc}",
        "",
        "---",
        f"## 各次探索结果（共 {n_trials} 次）",
    ]
    for i, ts in enumerate(trial_summaries):
        combo = ts.get("combo") or []
        weights = ts.get("weights") or {}
        bs = ts.get("backtest_stats") or {}
        metrics = ts.get("metrics") or {}
        ab = ts.get("alpha_beta") or {}
        lines.append(f"\n### Trial {i}")
        lines.append(f"- 因子组合：{combo}")
        lines.append(f"- 线性权重：{weights}")
        lines.append(f"- 年化夏普：{bs.get('sharpe_annual', '—')}  |  总收益：{round(float(bs.get('total_return', 0) or 0) * 100, 2)}%  |  最大回撤：{round(float(bs.get('max_drawdown', 0) or 0) * 100, 2)}%")
        lines.append(f"- Alpha(年化)：{ab.get('annualized_alpha', '—')}  |  Beta：{ab.get('beta', '—')}  |  R²：{ab.get('r_squared', '—')}")
        lines.append(f"- 验证集 IC：{metrics.get('val_ic', metrics.get('val_rank_ic', '—'))}  |  训练集 IC：{metrics.get('train_ic', metrics.get('train_rank_ic', '—'))}")
        for f in combo:
            base_f = f[:-5] if f.endswith("_bell") else f
            q = quality_per_factor.get(base_f) or quality_per_factor.get(f, {})
            if q:
                lines.append(f"  - {f}: spread={q.get('spread', '—')}, IC={q.get('ic', '—')}, IC_IR={q.get('ic_ir', '—')}, dir={q.get('direction', '—')}")
    if corr_text:
        lines.extend(["", "## 因子间相关系数矩阵", corr_text])
    lines.extend([
        "",
        "---",
        "## 要求",
        "1. 选出最优 trial（selected_trial_idx，0-indexed）。",
        "2. 输出 verdict、quality_score、reliability、cap_recommendation。",
        "3. 策略逻辑、轮仓逻辑、评审意见（comments）、风险提示（risks）。",
        "4. 评审意见中若引用数字，必须来自你选中的 Trial 的表格，不得使用其他 trial 的数值。",
        "",
        "**输出格式**：仅输出一个 JSON。",
        '{"selected_trial_idx": 0, "verdict": "推荐", "quality_score": 7.5, "reliability": "中", "cap_recommendation": "全市场", '
        '"strategy_logic": "...", "rotation_logic": "...", "comments": ["..."], "risks": ["..."]}',
    ])
    return "\n".join(line for line in lines)


def _extract_json_from_text(text: str) -> str:
    """从可能含 markdown 的回复中提取 JSON 字符串。"""
    if not text:
        return ""
    text = text.strip()
    if "```" in text:
        for marker in ("```json", "```"):
            if marker in text:
                start = text.find(marker) + len(marker)
                end = text.find("```", start)
                text = text[start: end if end != -1 else len(text)].strip()
                break
    return text


def parse_reviewer_phase1_json(text: str) -> Optional[Dict[str, Any]]:
    """解析 Phase1 输出：仅 selected_trial_idx, verdict, quality_score, reliability, cap_recommendation。"""
    text = _extract_json_from_text(text)
    if not text:
        return None
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        return {
            "selected_trial_idx": int(obj.get("selected_trial_idx", 0)),
            "verdict": str(obj.get("verdict", "谨慎推荐")),
            "quality_score": float(obj.get("quality_score", 5.0)),
            "reliability": str(obj.get("reliability", "中")),
            "cap_recommendation": str(obj.get("cap_recommendation") or "全市场"),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def parse_reviewer_phase2_json(text: str) -> Optional[Dict[str, Any]]:
    """解析 Phase2 输出：strategy_logic, rotation_logic, comments, risks。"""
    text = _extract_json_from_text(text)
    if not text:
        return None
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        return {
            "strategy_logic": str(obj.get("strategy_logic") or ""),
            "rotation_logic": str(obj.get("rotation_logic") or ""),
            "comments": list(obj.get("comments") or []),
            "risks": list(obj.get("risks") or []),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def parse_reviewer_json(text: str) -> Optional[Dict[str, Any]]:
    """从审查 Agent 单阶段回复中解析完整 JSON（兼容旧逻辑）。"""
    text = _extract_json_from_text(text)
    if not text:
        return None
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        return {
            "selected_trial_idx": int(obj.get("selected_trial_idx", 0)),
            "verdict": str(obj.get("verdict", "谨慎推荐")),
            "quality_score": float(obj.get("quality_score", 5.0)),
            "reliability": str(obj.get("reliability", "中")),
            "cap_recommendation": str(obj.get("cap_recommendation") or "全市场"),
            "comments": list(obj.get("comments") or []),
            "risks": list(obj.get("risks") or []),
            "strategy_logic": str(obj.get("strategy_logic") or ""),
            "rotation_logic": str(obj.get("rotation_logic") or ""),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
