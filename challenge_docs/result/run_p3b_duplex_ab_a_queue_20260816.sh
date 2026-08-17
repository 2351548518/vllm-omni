#!/usr/bin/env bash
set -o pipefail

# The P3B simplex and Seed-TTS gates must finish before exercising the
# full-duplex path.  A passing simplex candidate is the only case that pays
# the additional duplex cost; a failing candidate is recorded and skipped.
while tmux has-session -t minicpm_opt_p3b 2>/dev/null; do
  sleep 30
done

P3B_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_b_generation_simplex
mapfile -t P3B_JSONS < <(find "$P3B_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if [ "${#P3B_JSONS[@]}" -ne 3 ]; then
  echo "P3B_DUPLEX_SKIP=missing_three_simplex_jsons count=${#P3B_JSONS[@]}"
  exit 0
fi
for result in "${P3B_JSONS[@]}"; do
  if ! jq -e '.failed == 0 and .completed == .num_prompts' "$result" >/dev/null; then
    echo "P3B_DUPLEX_SKIP=simplex_request_failure file=$result"
    exit 0
  fi
done
P3B_ACC=$(find /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_generation_accuracy \
  -maxdepth 1 -type f -name '*.json' | sort | tail -1)
if [ -z "$P3B_ACC" ] || ! jq -e '.completed == .num_prompts and .failed == 0 and .seed_tts_request_failed == 0 and .seed_tts_no_pcm == 0 and .seed_tts_content_error_mean <= 0.02' "$P3B_ACC" >/dev/null; then
  echo "P3B_DUPLEX_SKIP=seed_tts_gate_failed_or_missing"
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
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_duplex_a1_default \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3b_clean_duplex_a1_default_20260816_143000.log
run_duplex /workspace/user_data/vllm-omni-worktrees/p3b-shm-generation \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_duplex_b_generation \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3b_clean_duplex_b_generation_20260816_143000.log
run_duplex /workspace/user_data/vllm-omni \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_duplex_a2_default \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3b_clean_duplex_a2_default_20260816_143000.log
