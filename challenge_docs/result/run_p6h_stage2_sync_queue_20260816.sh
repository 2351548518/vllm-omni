#!/usr/bin/env bash
set -o pipefail

while tmux has-session -t minicpm_opt_queue_p6g 2>/dev/null; do
  sleep 30
done

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
  for _ in $(seq 1 30); do pgrep -f 'vllm_omni.entrypoints.cli.main serve' >/dev/null || break; sleep 2; done
  find /dev/shm -maxdepth 1 -type f \( -name 'shm_*' -o -name 'chatcmpl-*' -o -name 'sem.*' \) -delete
}

run_perf /workspace/user_data/vllm-omni \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p6h_a1_default_simplex \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p6h_a1_default_simplex_20260816_120000.log
run_perf /workspace/user_data/vllm-omni-worktrees/p6h-stage2-sync \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p6h_b_stage2_sync_simplex \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p6h_b_stage2_sync_simplex_20260816_120000.log
run_perf /workspace/user_data/vllm-omni \
  /workspace/user_data/vllm-omni/challenge_docs/batch_result/p6h_a2_default_simplex \
  /workspace/user_data/vllm-omni/challenge_docs/run_log/p6h_a2_default_simplex_20260816_120000.log

cd /workspace/user_data/vllm-omni-worktrees/p6h-stage2-sync
export PYTHONPATH="/workspace/user_data/vllm-omni-worktrees/p6h-stage2-sync:${PYTHONPATH:-}"
env ASCEND_RT_VISIBLE_DEVICES=0 \
  VLLM_TEST_MINICPMO_4_5_MODEL=/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  VLLM_SEED_TTS_DATASET_PATH=/workspace/user_data/datasets/seed-tts-eval/seedtts_testset \
  ACC_BENCH_SEED_TTS_MAX_CONCURRENCY=4 SEED_TTS_WER_EVAL=1 SEED_TTS_SIM_EVAL=1 \
  ACC_BENCH_RESULT_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/p6h_stage2_sync_accuracy \
  pytest -s -v tests/e2e/accuracy/minicpmo_4_5/test_minicpmo_4_5.py \
  -m full_model -k 'seed_tts_wer_bench and not duplex' --run-level full_model 2>&1 | \
  tee /workspace/user_data/vllm-omni/challenge_docs/run_log/p6h_stage2_sync_seed_accuracy_20260816_120000.log
echo "SEED_RC=${PIPESTATUS[0]}"
