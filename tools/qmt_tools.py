"""
QMT 量化工具：re-export 自 modules.qmt，保持向后兼容。
业务实现已迁至 modules/qmt/。
"""
from modules.qmt import *  # noqa: F401, F403
