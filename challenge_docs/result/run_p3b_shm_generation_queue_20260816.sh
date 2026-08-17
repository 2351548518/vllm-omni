#!/usr/bin/env bash
set -o pipefail

while tmux has-session -t minicpm_opt_queue_p3 2>/dev/null; do
  sleep 30
done

P3_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/p3_b_shm_length_simplex
mapfile -t P3_JSONS < <(find "$P3_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if [ "${#P3_JSONS[@]}" -eq 3 ]; then
  p3_ok=1
  for result in "${P3_JSONS[@]}"; do
    if ! jq -e '.failed == 0 and .completed == .num_prompts' "$result" >/dev/null; then
      p3_ok=0
      break
    fi
  done
  P3_ACC=$(find /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3_shm_length_accuracy \
    -maxdepth 1 -type f -name '*.json' | sort | tail -1)
  if [ "$p3_ok" -eq 1 ] && [ -n "$P3_ACC" ]; then
    jq -e '.completed == .num_prompts and .failed == 0 and .seed_tts_request_failed == 0 and .seed_tts_no_pcm == 0 and .seed_tts_content_error_mean <= 0.02' "$P3_ACC" >/dev/null || p3_ok=0
  else
    p3_ok=0
  fi
  if [ "$p3_ok" -eq 1 ]; then
    echo "P3B_SKIP=p3_shm_length_and_seed_passed"
    exit 0
  fi
fi

run_perf() {
  local wd="$1" result_dir="$2" log="$3"
  cd "$wd"
  export PYTHONPATH="$wd:${PYTHONPATH:-}" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
  BENCHMARK_DIR="$result_dir" python -m pytest -s -v \
    tests/dfx/perf/scripts/run_benchmark.py \
    --test-config-file /workspace/user_data/vllm-omni/challenge_docs/test_yaml/test_minicpmo_4_5.json \
    -k test_minicpmo_4_5_challenge 2>&1 | tee "$log"
  echo "PERF_RC=${PIPESTATUS[0]}"
  # Chapter 10 shuts down the fixture service, but an abnormal stage exit can
  # leave named SHM lock/data files behind.  Wait for the service process to
  # disappear, then remove only the benchmark-owned ephemeral names before the
  # next A/B leg.  This keeps /dev/shm capacity from contaminating the result.
  for _ in $(seq 1 30); do
    pgrep -f 'vllm_omni.entrypoints.cli.main serve' >/dev/null || break
    sleep 2
  done
  find /dev/shm -maxdepth 1 -type f \( -name 'shm_*' -o -name 'chatcmpl-*' -o -name 'sem.*' \) -delete
}

run_perf /workspace/user_data/vllm-omni \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_a1_default_simplex \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3b_clean_a1_default_simplex_20260816_143000.log
run_perf /workspace/user_data/vllm-omni-worktrees/p3b-shm-generation \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_b_generation_simplex \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3b_clean_b_generation_simplex_20260816_143000.log
run_perf /workspace/user_data/vllm-omni \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_a2_default_simplex \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p3b_clean_a2_default_simplex_20260816_143000.log

cd /workspace/user_data/vllm-omni-worktrees/p3b-shm-generation
export PYTHONPATH="/workspace/user_data/vllm-omni-worktrees/p3b-shm-generation:${PYTHONPATH:-}"
env ASCEND_RT_VISIBLE_DEVICES=0 \
  VLLM_TEST_MINICPMO_4_5_MODEL=/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  VLLM_SEED_TTS_DATASET_PATH=/workspace/user_data/datasets/seed-tts-eval/seedtts_testset \
  ACC_BENCH_SEED_TTS_MAX_CONCURRENCY=4 SEED_TTS_WER_EVAL=1 SEED_TTS_SIM_EVAL=1 \
  ACC_BENCH_RESULT_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/p3b_clean_generation_accuracy \
  pytest -s -v tests/e2e/accuracy/minicpmo_4_5/test_minicpmo_4_5.py \
  -m full_model -k 'seed_tts_wer_bench and not duplex' --run-level full_model 2>&1 | \
  tee /workspace/user_data/vllm-omni/challenge_docs/run_log/p3b_clean_generation_seed_accuracy_20260816_143000.log
echo "SEED_RC=${PIPESTATUS[0]}"
