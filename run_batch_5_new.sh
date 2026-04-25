#!/usr/bin/env bash
# 5 只新标的 v3 批处理(含 Tushare 增强 + 真视觉)
set -e
cd "$(dirname "$0")"

# 从 Tushare 动态获取名字(避免 shell 中文乱码)
STOCKS=(
  "301667"
  "920111"
  "300890"
  "601860"
  "300260"
)

mkdir -p output/batch_logs
START_TS=$(date +%s)
TOTAL=${#STOCKS[@]}
i=0

for code in "${STOCKS[@]}"; do
  i=$((i + 1))
  echo ""
  echo "════════════════════════════════════════════════════"
  echo "[${i}/${TOTAL}] v3 跑 $code (全新采集+Tushare+真视觉)"
  echo "════════════════════════════════════════════════════"
  log="output/batch_logs/v3_new_${code}.log"
  # name 使用代码本身, 系统会自动从数据源获取真名
  python run.py analyze \
    --symbol "$code" --name "$code" --version v3 \
    --debate-rounds 3 --risk-rounds 2 \
    > "$log" 2>&1 || echo "FAIL: $code" | tee -a "$log"
  grep -E "^\[v3\] (run_dir|final_decision|duration)" "$log" | head -3
done

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
MIN=$((DURATION / 60))
echo ""
echo "════════════════════════════════════════════════════"
echo "5 只完成, 耗时 ${MIN} 分钟"
echo "════════════════════════════════════════════════════"

# 汇总
echo ""
echo "=== 结果 ==="
for code in "${STOCKS[@]}"; do
  log="output/batch_logs/v3_new_${code}.log"
  if [ -f "$log" ]; then
    score=$(grep -oE "score=[0-9.]+" "$log" | tail -1 | cut -d= -f2)
    dec=$(grep -oE "final_decision: [a-zA-Z_]+" "$log" | tail -1 | cut -d: -f2 | xargs)
    echo "  $code  score=${score:-N/A}  decision=${dec:-FAIL}"
  fi
done
