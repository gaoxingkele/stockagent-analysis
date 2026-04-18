#!/usr/bin/env bash
# V3 vs V2 对比批处理 - 24 只去重股票
# 每只复用 v2 已有的 analysis_context.json(避免重采集)
set -e
cd "$(dirname "$0")"

# 格式: "code name v2_run_dir"
STOCKS=(
  # 7 只养猪/农牧(含 603363)
  "300498 温氏股份 output/runs/20260418_142803_300498"
  "002714 牧原股份 output/runs/20260418_143434_002714"
  "002124 天邦食品 output/runs/20260418_144014_002124"
  "002157 正邦科技 output/runs/20260418_145037_002157"
  "000876 新希望 output/runs/20260418_150008_000876"
  "603477 巨星农牧 output/runs/20260418_151100_603477"
  "603363 傲农生物 output/runs/20260418_151800_603363"
  # 17 只杂股(去重 603363)
  "920047 诺思兰德 output/runs/20260417_154748_920047"
  "688175 高凌信息 output/runs/20260417_155417_688175"
  "688215 瑞晟智能 output/runs/20260417_160159_688215"
  "002571 德力股份 output/runs/20260417_161730_002571"
  "600388 龙净环保 output/runs/20260417_162431_600388"
  "301186 超达装备 output/runs/20260417_163209_301186"
  "002407 多氟多 output/runs/20260417_163844_002407"
  "002050 三花智控 output/runs/20260417_164603_002050"
  "600488 津药药业 output/runs/20260417_165249_600488"
  "301682 宏明电子 output/runs/20260417_165923_301682"
  "001257 盛龙股份 output/runs/20260417_170444_001257"
  "688805 健信超导 output/runs/20260417_171037_688805"
  "002832 比音勒芬 output/runs/20260417_171755_002832"
  "002183 怡亚通 output/runs/20260417_172542_002183"
  "002491 通鼎互联 output/runs/20260417_173313_002491"
  "001288 运机集团 output/runs/20260417_174136_001288"
  "301221 光庭信息 output/runs/20260417_174814_301221"
)

mkdir -p output/batch_logs

# v3 参数: 4 轮多空辩论 + 3 轮风控(去极端化 + 深辩论)
DEBATE_ROUNDS=4
RISK_ROUNDS=3

START_TS=$(date +%s)
TOTAL=${#STOCKS[@]}
i=0
for entry in "${STOCKS[@]}"; do
  i=$((i + 1))
  read -r code name run_dir <<< "$entry"
  echo "════════════════════════════════════════════════════"
  echo "[${i}/${TOTAL}] v3 开始: $code $name"
  echo "复用 context: $run_dir"
  echo "════════════════════════════════════════════════════"

  log="output/batch_logs/v3_${code}_${name}.log"
  python run.py analyze \
    --symbol "$code" --name "$name" --version v3 \
    --debate-rounds $DEBATE_ROUNDS --risk-rounds $RISK_ROUNDS \
    --run-dir "$run_dir" \
    > "$log" 2>&1 || echo "FAIL: $code $name" | tee -a "$log"

  # 提取本次结果
  final_line=$(grep -E "^\[v3\] final_decision|^\[v3\] run_dir|^\[v3\] duration" "$log" | head -3)
  echo "$final_line"
done

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
echo ""
echo "════════════════════════════════════════════════════"
echo "全部完成 ($(date +%Y-%m-%d' '%H:%M:%S), 耗时 ${DURATION}s)"
echo "════════════════════════════════════════════════════"

# 汇总 v3 结果
echo ""
echo "=== v3 结果汇总 ==="
printf "%-8s %-12s %-8s %-16s %-8s\n" "Code" "Name" "Score" "Decision" "Duration"
echo "------------------------------------------------------------"
for entry in "${STOCKS[@]}"; do
  read -r code name run_dir <<< "$entry"
  log="output/batch_logs/v3_${code}_${name}.log"
  if [ -f "$log" ]; then
    score=$(grep -oE "score=[0-9.]+" "$log" | tail -1 | cut -d= -f2)
    decision=$(grep -oE "final_decision: [a-zA-Z_]+" "$log" | tail -1 | cut -d: -f2 | xargs)
    dur=$(grep -oE "duration: [0-9.]+s" "$log" | tail -1 | cut -d: -f2 | xargs)
    printf "%-8s %-12s %-8s %-16s %-8s\n" "$code" "$name" "${score:-N/A}" "${decision:-FAIL}" "${dur:-N/A}"
  else
    printf "%-8s %-12s %-8s %-16s %-8s\n" "$code" "$name" "N/A" "NO_LOG" "N/A"
  fi
done
