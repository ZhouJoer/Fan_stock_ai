"""
因子挖掘模块入口。

提供因子定义、横截面评分与评估工具。
"""

from .factor_definitions import DEFAULT_FACTOR_GROUP_WEIGHTS, DEFAULT_FACTOR_WEIGHTS
from .factor_engine import score_cross_section
from .factor_eval import FactorEvaluator, compute_alpha_beta
from .factor_trainer import (
    TrainConfig,
    build_training_samples,
    train_linear_softmax_weights,
    apply_bell_transforms,
)
from .model_store import ensure_model_dir, save_factor_model, load_factor_model, list_factor_models, delete_factor_model
from .prompts import NODE_ROLES, get_agent_orchestration_prompt
from .workflow import factor_mining_graph

__all__ = [
    "DEFAULT_FACTOR_GROUP_WEIGHTS",
    "DEFAULT_FACTOR_WEIGHTS",
    "score_cross_section",
    "FactorEvaluator",
    "compute_alpha_beta",
    "TrainConfig",
    "build_training_samples",
    "train_linear_softmax_weights",
    "apply_bell_transforms",
    "ensure_model_dir",
    "save_factor_model",
    "load_factor_model",
    "list_factor_models",
    "delete_factor_model",
    "factor_mining_graph",
    "NODE_ROLES",
    "get_agent_orchestration_prompt",
]
