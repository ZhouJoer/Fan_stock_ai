"""多用户认证：JWT、登录、注册"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from db.users import create_user, delete_user, get_user_by_username, verify_password

router = APIRouter(prefix="/api", tags=["auth"])
security = HTTPBearer(auto_error=False)

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7  # 7 天
MULTI_USER_MODE = os.getenv("MULTI_USER_MODE", "false").lower() == "true"

# 单用户模式下的默认用户 ID
DEFAULT_USER_ID = "default"


class UserOut(BaseModel):
    user_id: str
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class DeleteAccountRequest(BaseModel):
    password: str


def _create_token(user_id: str, username: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    获取当前用户。单用户模式下始终返回 default 用户；
    多用户模式下需有效 JWT，否则 401。
    """
    if not MULTI_USER_MODE:
        return {"user_id": DEFAULT_USER_ID, "username": "default"}

    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="需要登录")

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
        user_id = payload.get("sub")
        username = payload.get("username", "")
        if not user_id:
            raise HTTPException(status_code=401, detail="无效的 token")
        return {"user_id": user_id, "username": username}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的 token")


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    可选认证：多用户模式下无 token 返回 None（由调用方决定是否 401）；
    单用户模式返回 default。
    """
    if not MULTI_USER_MODE:
        return {"user_id": DEFAULT_USER_ID, "username": "default"}

    if not credentials or not credentials.credentials:
        return None

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
        user_id = payload.get("sub")
        username = payload.get("username", "")
        if not user_id:
            return None
        return {"user_id": user_id, "username": username}
    except jwt.InvalidTokenError:
        return None


def get_user_data_dir(user_id: str) -> str:
    """
    返回用户数据根目录。
    单用户 default 使用项目根下的 pools/、data/ 等，保持兼容；
    多用户使用 data/users/{user_id}/ 下的子目录。
    """
    if user_id == DEFAULT_USER_ID:
        return ""
    return f"data/users/{user_id}"


def get_user_pools_dir(user_id: str) -> str:
    if user_id == DEFAULT_USER_ID:
        return "pools"
    return f"data/users/{user_id}/pools"


def get_user_pool_sim_dir(user_id: str) -> str:
    if user_id == DEFAULT_USER_ID:
        return "data/stock_pool_sim_accounts"
    return f"data/users/{user_id}/stock_pool_sim_accounts"


def get_user_etf_sim_dir(user_id: str) -> str:
    if user_id == DEFAULT_USER_ID:
        return "data/etf_sim_accounts"
    return f"data/users/{user_id}/etf_sim_accounts"


@router.post("/auth/register", response_model=dict)
async def register(req: RegisterRequest):
    """注册新用户"""
    if not MULTI_USER_MODE:
        raise HTTPException(status_code=400, detail="当前为单用户模式，无需注册")

    username = (req.username or "").strip()
    if not username or len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少 2 个字符")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")

    existing = get_user_by_username(username)
    if existing:
        raise HTTPException(status_code=400, detail="用户名已存在")

    user = create_user(username, req.password)
    token = _create_token(user["user_id"], username)
    return {
        "token": token,
        "user": {"user_id": user["user_id"], "username": username},
    }


@router.post("/auth/login", response_model=dict)
async def login(req: LoginRequest):
    """登录"""
    if not MULTI_USER_MODE:
        return {
            "token": "",
            "user": {"user_id": DEFAULT_USER_ID, "username": "default"},
        }

    user = get_user_by_username(req.username.strip())
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = _create_token(user["user_id"], user["username"])
    return {
        "token": token,
        "user": {"user_id": user["user_id"], "username": user["username"]},
    }


@router.get("/auth/me", response_model=dict)
async def auth_me(user: dict = Depends(get_current_user)):
    """获取当前登录用户"""
    return {"user_id": user["user_id"], "username": user["username"]}


@router.get("/auth/mode")
async def auth_mode():
    """查询是否为多用户模式"""
    return {"multi_user": MULTI_USER_MODE}


@router.post("/auth/account/delete")
async def delete_account(
    req: DeleteAccountRequest,
    user: dict = Depends(get_current_user),
):
    """注销当前账户，需验证密码。将删除账户及所有用户数据。"""
    if not MULTI_USER_MODE:
        raise HTTPException(status_code=400, detail="单用户模式无需注销")

    user_id = user.get("user_id")
    if user_id == DEFAULT_USER_ID:
        raise HTTPException(status_code=400, detail="默认用户无法注销")

    if not delete_user(user_id, req.password):
        raise HTTPException(status_code=401, detail="密码错误")

    import shutil
    user_data_dir = get_user_data_dir(user_id)
    if user_data_dir and os.path.exists(user_data_dir):
        try:
            shutil.rmtree(user_data_dir)
        except Exception as e:
            print(f"[auth] 删除用户数据目录失败: {e}")

    return {"ok": True, "message": "账户已注销"}
