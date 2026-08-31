# 会话交接 (SESSION HANDOFF) — 2026-06-24

> **最新 checkpoint（2026-08-31）**：当前工作已转入独立 S20 高置信上涨选股研究线。
> 恢复时优先阅读 [`CHECKPOINT_S20_20260831.md`](CHECKPOINT_S20_20260831.md)；本文件以下
> 2026-06-24 内容保留为旧生产/研究背景，不代表 S20 最新状态。

> 重开窗口后先读本文件 + 全局/项目 CLAUDE.md + `git log`，即可恢复。本会话跨越多个研究 campaign + web 重构。

## 0. 环境 (重开后关键)
- **Git 在 `D:\Program Files\Git`**。已设 `CLAUDE_CODE_GIT_BASH_PATH=D:\Program Files\Git\bin\bash.exe`(User级)；**重开窗口后 Bash 工具应自动生效**(本会话因进程先于设置启动而失效，全程用了 PowerShell)。
- 若 Bash 仍报 `Top-level not found: C:\Program Files\Git\bin` → 回退 PowerShell，每次命令前 `$env:PATH="D:\Program Files\Git\cmd;"+$env:PATH`。
- Python = `D:\Python314\python.exe`；中文乱码设 `$env:PYTHONIOENCODING="utf-8"`。
- Tushare token 在 `.env`(无 report_rc/news/anns_d 权限；有 fut_daily/index_daily/fund_*/major_news)。

## 1. 生产线状态
- **V12.31 全程冻结只读** (src/stockagent_analysis/v12_scoring.py + 生产模型)。本会话未动选股逻辑。
- 数据已更新到 **20260622**(最新交易日)。四池看板 CSV 在 `output/daily_pick/dashboard_20260622/`。
- 日更命令: `python daily_review.py` (自动检测最新交易日→补数据→全市场特征→生成四池, 约3-5分钟)。

## 2. 本会话研究结论 (全部 REJECT, 选股层已挖到底)
- **正交因子 campaign (0618)**: 低波动/隔夜真异象过Phase-1但walk-forward被V12.31吃; 商品塌缩; report_rc/news无权限. 净0新alpha. (research/orthogonal-alpha)
- **市场状态暴露 overlay (0618)**: paper"alpha熊市强"对long-only反指(我们alpha bull-loaded). REJECT. (research/market-state-exposure)
- **QuantML 4轴 (0619)**: 排序目标/NN模型/Alpha-Beta解耦/自动因子挖掘 全REJECT → 数据不行非模型不行. (research/quantml-models)
- **基金抱团→alpha 3方案 (0619)**: 抱团水平=动量镜像, 流量边缘, 池walk-forward REJECT. 但描述性(高收益基金抱团光模块,中际旭创94%/新易盛93%)有效. (research/fund-crowding)
- **V12 因子审计 (0624)**: TreeSHAP×消融 → market_context/moneyflow=承重墙, cyb_rel_strength高用可替代, 估值=冗余. research/V12_FACTOR_AUDIT.md + PDF.
- 元结论: **增量不在选股(价量/正交/模型/基金持仓/审计全证), 在 book/风控/部署 或 真新数据(微结构/分析师修正, token无权限)**。

## 3. Web 子系统 (本会话重构, 已上线本地)
- 路径 `web/` (FastAPI + htmx+alpine+tailwind+echarts, 端口 9000)。
- **改造**: V12页→**四池看板**(A系统/B自选/C基金重仓动态/D追高); **雪球浅色主题**(红涨绿跌); **四池升顶级菜单**(/v12?tab=X); **新建分析只在池B**, A/C/D系统自动; **首页重做**(积分左上+最近分析上移+四池完整列表); **登录预填**(18606099618/Ab18606099618 密码模式); **r20/ratio 与上日对比↑↓ + 表头点击排序**(r20/ratio/past5)。
- daily_dashboard.py 抽出 **build_pools()** (CLI+web单一真相) + 动态池C(被≥2高收益基金持有~81只) + 每日存 scores_<date>.parquet(供次日对比)。
- 池B自选清单(WATCHLISTS["B 自选"], 在 daily_dashboard.py): 含 002571/600388/300648/688783/300706/000962/300054/002842/688662/000733/**301027(华蓝)**/**603992(松霖)** 共12只。加自选=改此列表+重生成(或日后做web持久化)。
- **⚠ web服务器是后台进程, 关窗口即停**。重开后启动: `cd web; python -m uvicorn app.main:app --host 0.0.0.0 --port 9000` (Redis未连用in-memory broker, 不影响)。打开 http://localhost:9000/ 登录(已预填)→四池。

## 4. Git / 代码库状态
- 本会话工作分支 `research/fund-crowding` (含全部提交链)。
- **⚠ 推送踩坑**: `research/fund-crowding` 历史含 475MB 大文件(`research/cache/cb001/consolidation_panel.parquet`, 会话前35bf7d3误提)超GitHub 100MB → 无法推。
- **已解决**: 从干净 origin/main 新建 **`publish/web-fourpool`** 快照本会话全部代码(无大文件), **已推送 GitHub** (origin/publish/web-fourpool)。
- 5个未提交文件(update_factor_lab_from_tushare.py修改 + .omx/ + daily_top20_0604.py + update_features_to_0604/0605.py)是会话前就存在的, 非本会话, 未动。

## 5. 下一步 (按真实价值排序, 待用户定)
1. **★前向 paper-trade 攒数据**: V12.31-clean每日四池/picks持续落库, 攒真实前向P&L(唯一解"Sharpe1.31幅度宽带"). harness已建(DLV-002, research/cache/paper_trade)。
2. **★7月血洗窗复检(~2026-07)**: 0508-0603实盘窗满20d前向数据将到, 用research_env复检v3c动量血洗。
3. **WorldQuant BRAIN 自进化挖矿**: 唯一新矿(美股). 用户需注册账号(platform.worldquantbrain.com免费)+凭证存env(WQ_EMAIL/WQ_PASSWORD), 我搭harness。
4. 池B web持久化(免改代码加删自选)。
5. publish/web-fourpool 合并到 main / 或清大文件历史让原研究分支可推。

## 6. memory (跨会话, 已写)
关键 memory 在 `C:\Users\iamaf\.claude\projects\D--aicoding-stockagent-analysis\memory\`: project_orthogonal_alpha_campaign_0618 / project_market_state_exposure_reject_0618 / project_quantml_models_all_reject_0619 / project_fund_crowding_analysis_0619 / project_fund_crowding_alpha_reject_0619。MEMORY.md 是索引。
