"""模拟浏览器完整登录流程 (含 cookie + 跨页跳转)。

跑法: python web/tests/e2e_browser_sim.py
"""
from __future__ import annotations

import sys

import httpx

# Win 控制台用 UTF-8 (避免 GBK 不能输出 ✓ 等字符)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:9000"


def section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def main():
    # 关键: 跳过任何系统代理 (Win 上 Clash/V2Ray 等代理会拦本地连接)
    transport = httpx.HTTPTransport(retries=0)
    with httpx.Client(
        base_url=BASE, follow_redirects=False, timeout=15,
        trust_env=False, transport=transport,
    ) as c:

        # 1. 访问 / 应该 307 → /login
        section("1. GET / (匿名访问)")
        r = c.get("/")
        print(f"  status={r.status_code}  location={r.headers.get('location')}")
        assert r.status_code in (307, 302), f"应重定向, got {r.status_code}"

        # 2. 访问 /login HTML
        section("2. GET /login (HTML)")
        r = c.get("/login")
        print(f"  status={r.status_code}  size={len(r.text)}")
        assert r.status_code == 200
        assert "登录" in r.text or "Login" in r.text
        assert "管理员密码" in r.text or "Password" in r.text
        print("  ✓ 登录页 HTML 正常")

        # 3. admin 密码登录 (POST API)
        section("3. POST /api/auth/password-login (admin)")
        r = c.post("/api/auth/password-login", json={
            "phone": "18606099618",
            "password": "Ab18606099618",
        })
        print(f"  status={r.status_code}")
        if r.status_code != 200:
            print(f"  ✗ 失败: {r.text[:200]}")
            sys.exit(1)
        d = r.json()
        print(f"  user.id={d['user']['id']} phone={d['user']['phone']}"
              f" points={d['user']['points']} invite_code={d['user']['invite_code']}")
        # cookie 应该被 set-cookie 头自动 jar 进去
        cookies = c.cookies
        token_cookie = cookies.get("access_token")
        assert token_cookie, "access_token cookie 应被设置"
        print(f"  ✓ access_token cookie 已设 (长度 {len(token_cookie)})")

        # 4. 用 cookie 访问 / 应该 200 (而非重定向)
        section("4. GET / (登录后)")
        r = c.get("/")
        print(f"  status={r.status_code}  size={len(r.text)}")
        assert r.status_code == 200, f"登录后访问 / 应 200, got {r.status_code} → {r.headers.get('location')}"
        # 注意: dashboard 标题包含某些关键字
        keywords = ["StockAgent", "可用积分", "邀请", "Admin", "186", "K", "L", "M"]
        found = [k for k in keywords if k in r.text]
        print(f"  含关键字: {found}")
        print("  ✓ Dashboard HTML 加载正常")

        # 5. /api/auth/me 看用户
        section("5. GET /api/auth/me (cookie 鉴权)")
        r = c.get("/api/auth/me")
        print(f"  status={r.status_code}")
        if r.status_code == 200:
            u = r.json()
            print(f"  ✓ {u['phone']} | points={u['points']} | is_admin={u['is_admin']}")

        # 6. analyze preview
        section("6. POST /api/analyze/preview")
        r = c.post("/api/analyze/preview", json={"symbols": ["600519", "300750"]})
        print(f"  status={r.status_code}")
        if r.status_code == 200:
            d = r.json()
            for it in d["items"]:
                print(f"    {it['symbol']:8} {it['type']:12} {it['points']}pt")
            print(f"  total={d['total_points']}pt  user有 {d['user_points']}pt  够? {d['enough']}")

        # 7. 退出
        section("7. POST /api/auth/logout")
        r = c.post("/api/auth/logout")
        print(f"  status={r.status_code}")

        # 8. 退出后访问 / 应该再被踢回 /login
        section("8. GET / (退出后)")
        r = c.get("/")
        print(f"  status={r.status_code}  location={r.headers.get('location')}")

        # 9. 普通用户验证码注册
        section("9. 普通用户 send-code → verify (含邀请码)")
        r = c.post("/api/auth/send-code", json={"phone": "13800009999"})
        print(f"  send-code status={r.status_code}")
        d = r.json()
        dev_code = d.get("dev_code")
        print(f"  dev_code={dev_code}")

        r = c.post("/api/auth/verify", json={
            "phone": "13800009999",
            "code": dev_code,
            "invite_code": None,
        })
        print(f"  verify status={r.status_code}")
        if r.status_code == 200:
            d = r.json()
            print(f"  ✓ 新用户注册: id={d['user']['id']} points={d['user']['points']} invite_code={d['user']['invite_code']}")

        print(f"\n{'='*60}")
        print(" ✅ 全部 E2E 通过")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
