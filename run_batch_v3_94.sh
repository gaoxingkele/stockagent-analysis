#!/usr/bin/env bash
# 94 只全量 v3 批处理(含 PDF) - 复用 v2 context
set -e
cd "$(dirname "$0")"

POOL=stock_pool_94.txt
DEBATE_ROUNDS=4
RISK_ROUNDS=3

mkdir -p output/batch_logs

START_TS=$(date +%s)
TOTAL=$(wc -l < "$POOL" | tr -d ' ')
i=0

while IFS='|' read -r code name run_dir; do
  # 去除 CRLF 换行里混进来的 \r (stock_pool_94.txt 是 Windows 换行)
  code=$(echo -n "$code" | tr -d '\r')
  name=$(echo -n "$name" | tr -d '\r')
  run_dir=$(echo -n "$run_dir" | tr -d '\r')
  [ -z "$code" ] && continue
  i=$((i + 1))
  if [ "$run_dir" = "NO_CTX" ]; then
    run_dir_arg=""
    echo "[${i}/${TOTAL}] v3 全新采集: $code $name"
  else
    run_dir_arg="--run-dir $run_dir"
    echo "[${i}/${TOTAL}] v3 复用: $code $name (ctx=$(basename $run_dir))"
  fi

  log="output/batch_logs/v3_94_${code}_${name}.log"
  python run.py analyze \
    --symbol "$code" --name "$name" --version v3 \
    --debate-rounds $DEBATE_ROUNDS --risk-rounds $RISK_ROUNDS \
    $run_dir_arg \
    > "$log" 2>&1 || echo "FAIL: $code $name" | tee -a "$log"

  # 提取本次结果
  grep -E "^\[v3\] (final_decision|duration)" "$log" | head -2
done < "$POOL"

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
MINUTES=$((DURATION / 60))
echo ""
echo "════════════════════════════════════════════════════"
echo "全部完成 $(date +%Y-%m-%d' '%H:%M:%S), 耗时 ${MINUTES} 分钟"
echo "════════════════════════════════════════════════════"

# 统计
echo ""
echo "=== 结果统计 ==="
buy=0; hold=0; sell=0; fail=0
while IFS='|' read -r code name run_dir; do
  log="output/batch_logs/v3_94_${code}_${name}.log"
  if [ ! -f "$log" ]; then
    fail=$((fail + 1))
    continue
  fi
  dec=$(grep -oE "final_decision: [a-zA-Z_]+" "$log" | tail -1 | cut -d: -f2 | xargs)
  case "$dec" in
    buy)  buy=$((buy + 1)) ;;
    hold) hold=$((hold + 1)) ;;
    sell) sell=$((sell + 1)) ;;
    *)    fail=$((fail + 1)) ;;
  esac
done < "$POOL"
echo "BUY  : $buy"
echo "HOLD : $hold"
echo "SELL : $sell"
echo "FAIL : $fail"
