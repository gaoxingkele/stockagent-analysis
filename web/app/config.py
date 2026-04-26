"""应用配置 - 从 .env 加载, 通过 pydantic-settings 校验。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 (web/app/config.py → web/app → web → project_root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ROOT_ENV = _PROJECT_ROOT / ".env"
_ROOT_ENV_CLOUBIC = _PROJECT_ROOT / ".env.cloubic"

# 把根目录两个 .env 文件写入 os.environ, 供 stockagent_analysis 的
# os.getenv() 调用直接读取 (pydantic-settings 只填自身字段, 不写 os.environ)
# override=False: 系统已有的环境变量不被覆盖
load_dotenv(_ROOT_ENV, override=False)
load_dotenv(_ROOT_ENV_CLOUBIC, override=False)

# akshare/东方财富推送服务器走直连, 不经本地代理(避免 ProxyError)
import os as _os
_no_proxy = _os.getenv("NO_PROXY", "")
_eastmoney = "push2.eastmoney.com,push2his.eastmoney.com"
if _eastmoney not in _no_proxy:
    _os.environ["NO_PROXY"] = f"{_no_proxy},{_eastmoney}".strip(",")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # pydantic-settings 同步读一遍(供 settings.xxx 属性访问)
        env_file=[str(_ROOT_ENV), str(_ROOT_ENV_CLOUBIC), ".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === 应用 ===
    app_name: str = "StockAgent"
    app_env: Literal["development", "production"] = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production-32-chars-minimum"
    base_url: str = "http://localhost:9000"

    # === 服务器 ===
    host: str = "0.0.0.0"
    port: int = 9000

    # === 数据库 ===
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # === Redis ===
    redis_url: str = "redis://localhost:6379/0"

    # === 管理员 (锁定 18606099618) ===
    admin_phone: str = "18606099618"

    # === 积分 ===
    points_register_bonus: int = 100
    points_invite_new_user: int = 50
    points_invite_referrer: int = 100
    points_analyze_full_cost: int = 20            # LLM 全量评分(首次)
    points_analyze_full_cache_hit: int = 10       # LLM 全量当日命中缓存
    points_analyze_quant_cost: int = 1            # 量化评分(默认跟踪)
    quant_cache_same_day: bool = True             # 同日同股 quant 复用结果(仍扣 1pt)

    # === 验证码 ===
    sms_provider: Literal["mock", "aliyun", "twilio"] = "mock"
    sms_test_code: str = "8888"
    sms_code_ttl_seconds: int = 300
    sms_rate_limit_per_day: int = 5

    # === LLM / Tushare (复用现有 .env, 不强制存在) ===
    tushare_token: str = ""
    kimi_api_key: str = ""
    grok_api_key: str = ""
    doubao_api_key: str = ""
    deepseek_api_key: str = ""

    # === JWT ===
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 120
    jwt_refresh_token_expire_days: int = 7
    jwt_remember_me_days: int = 30          # "保持登录"勾选后的 token + cookie 有效期

    # === 健康检查定时 ===
    healthcheck_cron_enabled: bool = True
    healthcheck_cron_hours: str = "9-16"      # cron 小时表达式
    healthcheck_tz: str = "Asia/Shanghai"

    # === 订阅自动跟踪 (P10) ===
    subscription_cron_enabled: bool = True
    subscription_cron_time: str = "16:30"     # 每个交易日 16:30
    subscription_default_threshold: int = 5   # final_score 变化阈值

    # === 推送渠道 (P11) ===
    feishu_webhook_default: str = ""
    dingtalk_webhook_default: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    smtp_from: str = ""

    # === 日志 ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = "./logs"
    log_max_bytes: int = 50 * 1024 * 1024
    log_backup_count: int = 7

    # === 国际化 ===
    default_language: str = "zh-CN"
    supported_languages: str = "zh-CN,en-US,zh-TW"

    # === 限流 ===
    register_rate_limit_per_ip_day: int = 5
    analyze_queue_per_user: int = 1

    # === 路径计算属性 ===
    @property
    def project_root(self) -> Path:
        """stockagent-analysis 项目根目录(web 的上一级)。"""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def web_root(self) -> Path:
        """web/ 目录。"""
        return Path(__file__).resolve().parent.parent

    @property
    def runs_v3_dir(self) -> Path:
        """复用现有 output/runs_v3/ 目录。"""
        return self.project_root / "output" / "runs_v3"

    @property
    def supported_lang_list(self) -> list[str]:
        return [s.strip() for s in self.supported_languages.split(",") if s.strip()]


# 单例
settings = Settings()
