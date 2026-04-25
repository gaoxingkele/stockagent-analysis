#!/usr/bin/env bash
# 3 只新标的 v3.1 批处理 (含 Tushare 主源 + 真视觉 + quant_score)
set -e
cd "$(dirname "$0")"

STOCKS=(
  "600734"
  "600126"
  "688809"
)

mkdir -p output/batch_logs
START_TS=$(date +%s)
TOTAL=${#STOCKS[@]}
i=0

for code in "${STOCKS[@]}"; do
  i=$((i + 1))
  echo ""
  echo "════════════════════════════════════════════════════"
  echo "[${i}/${TOTAL}] v3.1 跑 $code (Tushare主源+quant_score)"
  echo "════════════════════════════════════════════════════"
  log="output/batch_logs/v3_new_${code}.log"
  python run.py analyze \
    --symbol "$code" --name "$code" --version v3 \
    --debate-rounds 4 --risk-rounds 3 \
    > "$log" 2>&1 || echo "FAIL: $code" | tee -a "$log"
  grep -E "^\[v3\] (run_dir|final_decision|duration)|final_score|decision_level" "$log" | head -5
done

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
MIN=$((DURATION / 60))
echo ""
echo "════════════════════════════════════════════════════"
echo "3 只完成, 耗时 ${MIN} 分钟"
echo "════════════════════════════════════════════════════"

echo ""
echo "=== 结果 ==="
for code in "${STOCKS[@]}"; do
  log="output/batch_logs/v3_new_${code}.log"
  if [ -f "$log" ]; then
    score=$(grep -oE "final_score['\"]?[:=][ ]*[0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
    dec=$(grep -oE "decision_level['\"]?[:=][ ]*['\"]?[a-z_]+" "$log" | tail -1 | grep -oE "[a-z_]+$")
    echo "  $code  score=${score:-N/A}  level=${dec:-FAIL}"
  fi
done
