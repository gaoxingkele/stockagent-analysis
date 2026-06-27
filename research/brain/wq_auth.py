# -*- coding: utf-8 -*-
"""WorldQuant BRAIN 认证 harness (I5 第一步).

BRAIN **没有静态 API key**: 程序化访问用注册邮箱+密码走 Basic Auth, POST /authentication 换 session cookie,
后续请求带这个 session。本模块: 读 .env (WQ_EMAIL/WQ_PASSWORD) → 登录 → 处理生物认证 → 验证 → 缓存 session。

用法:
  1. 在项目 .env 加: WQ_EMAIL=... / WQ_PASSWORD=...
  2. python research/brain/wq_auth.py        # 登录 + 验证 + 打印账户/权限

API:
  https://api.worldquantbrain.com/authentication   POST(Basic auth)→201 set-cookie t / 401 错或需生物认证
  https://api.worldquantbrain.com/users/self       GET→当前用户 (验证 session)

session 缓存到 research/brain/.wq_session.json (gitignored), 后续脚本 load_session() 复用, 过期自动重登。
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API = "https://api.worldquantbrain.com"
ROOT = Path(__file__).resolve().parents[2]
SESSION_FILE = Path(__file__).resolve().parent / ".wq_session.json"


def _creds() -> tuple[str, str]:
    email = os.getenv("WQ_EMAIL", "").strip()
    pwd = os.getenv("WQ_PASSWORD", "").strip()
    if not email or not pwd:
        sys.exit("缺凭证: 在项目 .env 加 WQ_EMAIL=... / WQ_PASSWORD=... (BRAIN 登录账密, 无独立 key)")
    return email, pwd


def sign_in(verbose: bool = True) -> requests.Session:
    """登录 → requests.Session (带 session cookie)。处理生物认证 (persona) 情况。"""
    email, pwd = _creds()
    s = requests.Session()
    s.auth = (email, pwd)
    r = s.post(f"{API}/authentication")
    if r.status_code == requests.codes.unauthorized:
        auth_hdr = r.headers.get("WWW-Authenticate", "")
        if "persona" in auth_hdr.lower():
            bio_url = urljoin(r.url, r.headers.get("Location", ""))
            print("⚠ 账户需生物认证 (persona)。浏览器打开完成认证:\n   " + bio_url)
            input("   完成后回车继续 ...")
            s.post(bio_url)
            r = s.post(f"{API}/authentication")
        else:
            sys.exit(f"登录失败 401 (账密错或账户限制): {r.text[:200]}")
    if r.status_code not in (200, 201):
        sys.exit(f"登录异常 HTTP {r.status_code}: {r.text[:200]}")
    if verbose:
        print(f"✓ 登录成功 HTTP {r.status_code} (session cookie 已获)")
    _save_session(s)
    return s


def _save_session(s: requests.Session):
    SESSION_FILE.write_text(json.dumps({
        "cookies": s.cookies.get_dict(), "ts": int(time.time()),
    }), encoding="utf-8")


def load_session() -> requests.Session:
    """复用缓存 session; 无效/过期则重登。供后续 BRAIN 脚本调用。"""
    s = requests.Session()
    if SESSION_FILE.exists():
        try:
            d = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            s.cookies.update(d.get("cookies", {}))
            if s.get(f"{API}/users/self").status_code == 200:
                return s
        except Exception:
            pass
    return sign_in(verbose=False)


def safe_req(s: requests.Session, method: str, url, retries: int = 6, **kw):
    """带重试的请求 (抗瞬时 RemoteDisconnected/Timeout/5xx + session 过期重登)。
    返回 response。网络抖动指数退避; 401 自动重登一次。供长跑 miner 用。"""
    last = None
    for i in range(retries):
        try:
            r = s.request(method, url, timeout=kw.pop("timeout", 30), **kw)
            if r.status_code == 401 and i == 0:
                ns = sign_in(verbose=False)
                s.cookies.update(ns.cookies.get_dict())
                continue
            if r.status_code >= 500:
                time.sleep(3 + 2 * i); last = r; continue
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last = e
            time.sleep(3 + 3 * i)
    if isinstance(last, requests.Response):
        return last
    raise last if last else RuntimeError(f"safe_req 失败 {method} {url}")


def whoami(s: requests.Session) -> dict:
    """拉当前用户 + 可用权限 (验证 session 有效)。"""
    r = s.get(f"{API}/users/self")
    r.raise_for_status()
    return r.json()


def main():
    print("=== WorldQuant BRAIN 认证 harness ===\n")
    s = sign_in()
    me = whoami(s)
    print("\n[账户]")
    for k in ("id", "email", "type"):
        if k in me:
            print(f"  {k}: {me[k]}")
    # 权限/层级 (字段名因账户而异, 全量打印关键块)
    for blk in ("permissions", "competition", "tier"):
        if blk in me:
            print(f"  {blk}: {json.dumps(me[blk], ensure_ascii=False)[:300]}")
    print(f"\n[session] 缓存 -> {SESSION_FILE.relative_to(ROOT)} (后续脚本 load_session() 复用)")
    print("[next] 认证通了 → 我接着搭: 拉数据字段 / 提交 alpha 表达式 / 读 IS·OS 指标的挖矿 harness")


if __name__ == "__main__":
    main()
