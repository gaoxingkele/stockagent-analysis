#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

STOCKS=(
  "600488 津药药业"
  "301682 宏明电子"
  "001257 盛龙股份"
  "688805 健信超导"
  "301186 超达装备"
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
    score=$(grep -E "final_decision|\[市场环境\]|\[评分\] 最终|\[乖离率\]" "$log" | tail -4)
    echo "--- $code $name ---"
    echo "$score"
  fi
done
