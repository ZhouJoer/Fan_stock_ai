"""
学习权重模型存储工具。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def ensure_model_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _safe_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        s = f"factor_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, "_")
    return s


def save_factor_model(model: Dict[str, object], directory: str, model_name: str = "") -> Dict[str, str]:
    ensure_model_dir(directory)
    name = _safe_name(model_name)
    filepath = os.path.join(directory, f"{name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    return {"model_name": name, "filepath": filepath}


def list_factor_models(directory: str) -> List[Dict[str, object]]:
    if not os.path.exists(directory):
        return []
    items = []
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items.append(
                {
                    "model_name": filename[:-5],
                    "factor_set": data.get("factor_set", "hybrid"),
                    "metrics": data.get("metrics", {}),
                    "created_at": data.get("created_at", ""),
                    "sample_count": data.get("sample_count", 0),
                    "filepath": path,
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def load_factor_model(directory: str, model_name: str = "") -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not os.path.exists(directory):
        return None, None
    if model_name:
        path = os.path.join(directory, f"{model_name}.json")
        if not os.path.exists(path):
            return None, None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), model_name

    models = list_factor_models(directory)
    if not models:
        return None, None
    latest = models[0]["model_name"]
    path = os.path.join(directory, f"{latest}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), latest


def delete_factor_model(directory: str, model_name: str) -> bool:
    if not model_name:
        return False
    path = os.path.join(directory, f"{_safe_name(model_name)}.json")
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True
