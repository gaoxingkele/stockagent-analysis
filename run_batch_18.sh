#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

STOCKS=(
  "920047 诺思兰德"
  "688175 高凌信息"
  "688215 瑞晟智能"
  "603363 傲农生物"
  "002571 德力股份"
  "600388 龙净环保"
  "301186 超达装备"
  "002407 多氟多"
  "002050 三花智控"
  "600488 津药药业"
  "301682 宏明电子"
  "001257 盛龙股份"
  "688805 健信超导"
  "002832 比音勒芬"
  "002183 怡亚通"
  "002491 通鼎互联"
  "001288 运机集团"
  "301221 光庭信息"
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
