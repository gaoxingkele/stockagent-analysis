#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

STOCKS=(
  "300498 温氏股份"
  "002714 牧原股份"
  "002124 天邦食品"
  "002157 正邦科技"
  "000876 新希望"
  "603477 巨星农牧"
  "603363 傲农生物"
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
    score=$(grep -E "final_decision|\[市场环境\]|\[评分\] 最终|\[乖离率\]|\[稀疏\]" "$log" | tail -5)
    echo "--- $code $name ---"
    echo "$score"
  fi
done
