#!/usr/bin/env bash
set -o pipefail

# P3 simplex 队列结束后才进入 full-duplex 门禁。若 P3 B 的三档普通流式
# 性能已有请求失败，直接留档并跳过双工长测，避免把无效候选继续扩大成本。
while tmux has-session -t minicpm_opt_queue_p3 2>/dev/null; do
  sleep 30
done

P3_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/p3_b_shm_length_simplex
mapfile -t P3_JSONS < <(find "$P3_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if [ "${#P3_JSONS[@]}" -ne 3 ]; then
  echo "P3_DUPLEX_SKIP=missing_three_simplex_jsons count=${#P3_JSONS[@]}"
  exit 0
fi
for result in "${P3_JSONS[@]}"; do
  if ! jq -e '.failed == 0 and .completed == .num_prompts' "$result" >/dev/null; then
    echo "P3_DUPLEX_SKIP=simplex_request_failure file=$result"
    exit 0
  fi
done
P3_ACC=$(find /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3_shm_length_accuracy \
  -maxdepth 1 -type f -name '*.json' | sort | tail -1)
if [ -z "$P3_ACC" ] || ! jq -e '.completed == .num_prompts and .failed == 0 and .seed_tts_request_failed == 0 and .seed_tts_no_pcm == 0 and .seed_tts_content_error_mean <= 0.02' "$P3_ACC" >/dev/null; then
  echo "P3_DUPLEX_SKIP=seed_tts_gate_failed_or_missing"
  exit 0
fi

run_duplex() {
  local wd="$1" result_dir="$2" log="$3"
  cd "$wd"
  export PYTHONPATH="$wd:${PYTHONPATH:-}" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
  BENCHMARK_DIR="$result_dir" pytest -s -v \
    tests/dfx/perf/scripts/run_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_minicpmo_4_5_duplex_seed_tts.json \
    2>&1 | tee "$log"
  echo "DUPLEX_RC=${PIPESTATUS[0]}"
}

run_duplex /workspace/user_data/vllm-omni \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3_duplex_a1_default \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3_duplex_a1_default_20260816_120000.log
run_duplex /workspace/user_data/vllm-omni-worktrees/p3-shm-length \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3_duplex_b_shm_length \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3_duplex_b_shm_length_20260816_120000.log
run_duplex /workspace/user_data/vllm-omni \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3_duplex_a2_default \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3_duplex_a2_default_20260816_120000.log
