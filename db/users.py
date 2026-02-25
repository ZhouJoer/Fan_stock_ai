"""用户表与 CRUD"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import bcrypt

USERS_DB = os.path.join(os.path.dirname(__file__), "users.json")


def _normalize_password(password) -> bytes:
    """统一密码格式，避免类型/空白导致验证失败"""
    s = str(password).strip() if password is not None else ""
    return s.encode("utf-8")[:72]


def _hash_password(password) -> str:
    pwd = _normalize_password(password)
    return bcrypt.hashpw(pwd, bcrypt.gensalt()).decode("ascii")


def _verify_password(password, hashed: str) -> bool:
    if not hashed or not isinstance(hashed, str):
        return False
    try:
        pwd = _normalize_password(password)
        hashed_bytes = hashed.encode("ascii") if isinstance(hashed, str) else hashed
        return bcrypt.checkpw(pwd, hashed_bytes)
    except Exception:
        return False
# 简单 JSON 存储，便于部署；可后续迁至 SQLite


def _load_users() -> list:
    if not os.path.exists(USERS_DB):
        return []
    try:
        with open(USERS_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_users(users: list):
    Path(USERS_DB).parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_DB, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def get_user_by_username(username: str) -> dict | None:
    users = _load_users()
    name_lower = username.strip().lower()
    for u in users:
        if u.get("username", "").lower() == name_lower:
            return u
    return None


def get_user_by_id(user_id: str) -> dict | None:
    users = _load_users()
    for u in users:
        if u.get("user_id") == user_id:
            return u
    return None


def create_user(username: str, password: str) -> dict:
    if get_user_by_username(username):
        raise ValueError("用户名已存在")

    user_id = str(uuid.uuid4())
    password_hash = _hash_password(password)
    user = {
        "user_id": user_id,
        "username": username.strip(),
        "password_hash": password_hash,
    }
    users = _load_users()
    users.append(user)
    _save_users(users)
    return {"user_id": user_id, "username": user["username"]}


def verify_password(password: str, password_hash: str) -> bool:
    return _verify_password(password, password_hash)


def delete_user(user_id: str, password: str) -> bool:
    """删除用户，需验证密码。返回是否成功。"""
    user = get_user_by_id(user_id)
    if not user:
        return False
    if not verify_password(password, user.get("password_hash", "")):
        return False
    users = _load_users()
    users = [u for u in users if u.get("user_id") != user_id]
    _save_users(users)
    return True
