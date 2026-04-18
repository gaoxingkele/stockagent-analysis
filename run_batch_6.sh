#!/usr/bin/env bash
# 批量跑剩余5只股票(002050已跑完)
set -e
cd "$(dirname "$0")"

STOCKS=(
  "920047 诺思兰德"
  "688175 高凌信息"
  "688215 瑞晟智能"
  "603363 傲农生物"
  "601866 中远海发"
)

mkdir -p output/batch_logs

for entry in "${STOCKS[@]}"; do
  code="${entry%% *}"
  name="${entry##* }"
  echo "========================================"
  echo "开始: $code $name"
  echo "========================================"
  python run.py analyze --symbol "$code" --name "$name" \
    > "output/batch_logs/${code}_${name}.log" 2>&1 || echo "FAIL: $code"
  echo "完成: $code"
done

echo ""
echo "=== 全部完成 ==="
echo "== 汇总 =="
for entry in "${STOCKS[@]}"; do
  code="${entry%% *}"
  name="${entry##* }"
  log="output/batch_logs/${code}_${name}.log"
  if [ -f "$log" ]; then
    score=$(grep -E "final_decision|\[市场环境\]|\[评分\] 最终" "$log" | tail -3)
    echo "--- $code $name ---"
    echo "$score"
  fi
done
