# -*- coding: utf-8 -*-
"""梅花易数 起卦特征编码器 — (ts_code, date, close) → 卦象 ML 特征。

梅花核心算法 vendored from D:/aicoding/ds-oracle-cli/app/engine/meihua.py
(TRIGRAMS / 体用五行关系 / 数字起卦 上卦%8 下卦%8 动爻%6), 纯 Python 无依赖。

固定起卦模式 (确定性, causal, 回测=实盘逐位一致):
  静态(代码) + 动态(公历日期 + 收盘价) 组合 → 上卦/下卦/动爻。
  另出一个纯代码"静态卦"作股票 ID 特征 (每股恒定)。
  +20 等 forward label 绝不进此编码 (只输入决策时刻已知量: 代码/日期/当日收盘价)。

输出特征 (每样本):
  组合卦 mh_*: base/mutual/changed gua id (1..64), up/lo trigram (1..8), moving (1..6),
              ti/yong 五行 (1..5), 体用关系 (0..4), 本卦阳爻数 (0..6)
  静态卦 mhs_*: 同上但仅由代码生成 (per-stock 常数 = 卦象版股票 embedding)

诚实定位: 这是一个确定性 categorical 特征变换。是否有信号由 walk-forward + 冻结 gate +
  横截面分位 + 朴素消融(农历月+板块 cohort)裁决, 不靠人工断语。先验=大概率噪声。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# ───────── 梅花核心 (vendored from ds-oracle-cli/app/engine/meihua.py) ─────────
# idx -> (name, 五行 element, lines 下->上 三爻 bit)
_TRI = {
    1: ("乾", "金", (1, 1, 1)), 2: ("兑", "金", (1, 1, 0)),
    3: ("离", "火", (1, 0, 1)), 4: ("震", "木", (1, 0, 0)),
    5: ("巽", "木", (0, 1, 1)), 6: ("坎", "水", (0, 1, 0)),
    7: ("艮", "土", (0, 0, 1)), 8: ("坤", "土", (0, 0, 0)),
}
_BY_LINES = {v[2]: k for k, v in _TRI.items()}          # lines -> idx
_WUXING = {
    "金": {"金": "比和", "木": "体克用", "水": "体生用", "火": "用克体", "土": "用生体"},
    "木": {"木": "比和", "土": "体克用", "火": "体生用", "金": "用克体", "水": "用生体"},
    "水": {"水": "比和", "火": "体克用", "木": "体生用", "土": "用克体", "金": "用生体"},
    "火": {"火": "比和", "金": "体克用", "土": "体生用", "水": "用克体", "木": "用生体"},
    "土": {"土": "比和", "水": "体克用", "金": "体生用", "木": "用克体", "火": "用生体"},
}
_ELEM_ID = {"金": 1, "木": 2, "水": 3, "火": 4, "土": 5}
_REL_ID = {"比和": 0, "用生体": 1, "体生用": 2, "用克体": 3, "体克用": 4}


def _cast(up_num: int, lo_num: int, mv_num: int) -> dict:
    """数字起卦内核: 上卦=up_num%8, 下卦=lo_num%8, 动爻=mv_num%6 (余0进位)."""
    up = up_num % 8 or 8
    lo = lo_num % 8 or 8
    mv = mv_num % 6 or 6
    up_lines, lo_lines = _TRI[up][2], _TRI[lo][2]
    base = list(lo_lines + up_lines)                    # 6 爻, 下->上
    hu_lo = _BY_LINES[tuple(base[1:4])]                 # 互卦下 = 2,3,4 爻
    hu_up = _BY_LINES[tuple(base[2:5])]                 # 互卦上 = 3,4,5 爻
    chg = base.copy(); chg[mv - 1] = 1 - chg[mv - 1]    # 变卦: 翻动爻
    chg_lo = _BY_LINES[tuple(chg[:3])]; chg_up = _BY_LINES[tuple(chg[3:])]
    # 体用: 动爻在下卦(1-3)则下卦为用上卦为体; 在上卦(4-6)则上卦为用下卦为体
    ti_idx, yong_idx = (lo, up) if mv > 3 else (up, lo)
    ti_e, yong_e = _TRI[ti_idx][1], _TRI[yong_idx][1]
    rel = _WUXING[ti_e][yong_e]
    gid = lambda u, l: (u - 1) * 8 + l                  # 1..64
    return {
        "base_gua": gid(up, lo), "mutual_gua": gid(hu_up, hu_lo),
        "changed_gua": gid(chg_up, chg_lo),
        "upper": up, "lower": lo, "moving": mv,
        "ti_elem": _ELEM_ID[ti_e], "yong_elem": _ELEM_ID[yong_e],
        "relation": _REL_ID[rel], "yang_count": sum(base),
    }


def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(int(n))))


# ───────── 固定起卦模式 ─────────
def meihua_features(ts_code: str, date_str: str, close: float) -> dict:
    """date_str: 'YYYYMMDD' 公历; close: 当日收盘价. 返回 mh_* + mhs_* 特征 dict."""
    code6 = "".join(ch for ch in ts_code if ch.isdigit())[:6] or "0"
    code_int = int(code6)
    code_sum = _digit_sum(code_int)
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    hour = 15                                            # 固定收盘时辰 (申时附近)
    cents = int(round(float(close) * 100))
    p_tail = cents % 100
    p_sum = _digit_sum(cents)

    # 组合卦: 体侧偏 静态代码+日期, 用侧偏 动态价格
    comb = _cast(up_num=code_sum + y + m + d,
                 lo_num=d + hour + p_tail + p_sum,
                 mv_num=code_sum + y + m + d + p_tail + p_sum)
    # 静态卦: 纯代码 (每股恒定 = 卦象版股票 ID)
    stat = _cast(up_num=_digit_sum(code6[:3]), lo_num=_digit_sum(code6[3:]),
                 mv_num=code_int)
    out = {f"mh_{k}": v for k, v in comb.items()}
    out.update({f"mhs_{k}": v for k, v in stat.items()})
    return out


def encode_frame(df: pd.DataFrame, code_col="ts_code", date_col="trade_date",
                 close_col="close") -> pd.DataFrame:
    """批量: df 需含 ts_code/trade_date/close, 返回原 df + 梅花特征列。"""
    recs = [meihua_features(r[code_col], str(r[date_col]), r[close_col])
            for r in df[[code_col, date_col, close_col]].to_dict("records")]
    feat = pd.DataFrame(recs, index=df.index)
    return pd.concat([df, feat], axis=1)


# ───────── 自检 / demo ─────────
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    # 1) worked example
    ex = meihua_features("600519.SH", "20260601", 1688.00)
    print("=== 例: 600519.SH @ 20260601 close=1688.00 ===")
    inv_e = {v: k for k, v in _ELEM_ID.items()}; inv_r = {v: k for k, v in _REL_ID.items()}
    print(f"  组合卦 base={ex['mh_base_gua']} 互={ex['mh_mutual_gua']} 变={ex['mh_changed_gua']} "
          f"动爻={ex['mh_moving']} 体五行={inv_e[ex['mh_ti_elem']]} 用五行={inv_e[ex['mh_yong_elem']]} "
          f"关系={inv_r[ex['mh_relation']]} 阳爻={ex['mh_yang_count']}")
    print(f"  静态卦 base={ex['mhs_base_gua']} (该股恒定)")

    # 2) 全市场截面分布 (20260601)
    p = ROOT / "output/tushare_cache/daily/20260601.parquet"
    if p.exists():
        d = pd.read_parquet(p)[["ts_code", "close"]].copy()
        d["trade_date"] = "20260601"
        enc = encode_frame(d)
        print(f"\n=== 20260601 全市场 {len(enc)} 股 卦象分布自检 ===")
        print("  组合卦 体用关系占比:")
        vc = enc["mh_relation"].map(inv_r).value_counts(normalize=True)
        for k, v in vc.items():
            print(f"    {k}: {v:.1%}")
        print(f"  组合卦 base_gua 唯一值: {enc['mh_base_gua'].nunique()}/64  "
              f"动爻分布: {dict(enc['mh_moving'].value_counts().sort_index())}")
        print(f"  静态卦 base_gua 唯一值: {enc['mhs_base_gua'].nunique()}/64 (按代码)")
        # 确定性自检: 同输入重算一致
        enc2 = encode_frame(d)
        same = (enc.filter(like="mh").values == enc2.filter(like="mh").values).all()
        print(f"  确定性自检 (重算一致): {same}")
