# 实验 worktree 索引

每个长测候选均保留独立分支和 worktree，避免后续恢复配置时覆盖实际测试代码。结果日志仍按实验名写入 `challenge_docs/run_log/` 和 `challenge_docs/batch_result/`。

| 实验 | 分支 | worktree | 固定提交 | 变量/状态 |
|---|---|---|---|---|
| P1 clone-view | `exp/p1-clone-view` | `/workspace/user_data/vllm-omni-worktrees/p1-clone-view` | `46b99f7d` | Code2Wav 两处 request cache clone→detached view；单次性能/Seed 较好，但复测 A1 与 B 在 8/128 均遇 SHM trailing bytes，A2/Seed 待完成，暂不通过 |
| P6-A n3 | `exp/p6a-n3` | `/workspace/user_data/vllm-omni-worktrees/p6a-n3` | `b65a12e9`；tag `exp/p6a-n3-20260816` | `token2wav_n_timesteps=3`；性能改善但 Seed WER/SIM 回退，拒绝；代码快照已独立保存 |
| P6-B float16 | `exp/p6b-float16` | `/workspace/user_data/vllm-omni-worktrees/p6b-float16` | `c287518b` | `token2wav_float16=true`；dtype mismatch，已拒绝 |
| P6-C chunk20 | `exp/p6c-chunk20` | `/workspace/user_data/vllm-omni-worktrees/p6c-chunk20` | `0a61a41d` | `codec_chunk_frames=20`；性能/Seed 完成，轻微退化，拒绝 |
| P6-C chunk30 | `exp/p6c-chunk30` | `/workspace/user_data/vllm-omni-worktrees/p6c-chunk30` | `33ddf4ce` | `codec_chunk_frames=30`；性能改善但 Seed 触发 SHM 崩溃，拒绝 |
| P6-D context0 | `exp/p6d-context0` | `/workspace/user_data/vllm-omni-worktrees/p6d-context0` | `ec04185f` | `codec_left_context_frames=0`；性能退化且 Seed SHM 崩溃，拒绝 |
| P6-D context6 | `exp/p6d-context6` | `/workspace/user_data/vllm-omni-worktrees/p6d-context6` | `c61778fb` | `codec_left_context_frames=6`；8/128 SHM 崩溃，Seed 137/2020，拒绝 |
| P6-E stage2-seqs8 | `exp/p6e-stage2-seqs8` | `/workspace/user_data/vllm-omni-worktrees/p6e-stage2-seqs8` | `699c1c99` | Stage 2 `max_num_seqs=8`；A/B/A+Seed 已排队 |
| P3 shm-length | `exp/p3-shm-length` | `/workspace/user_data/vllm-omni-worktrees/p3-shm-length` | `ae707303` | SHM magic+payload length framing；单测断言通过但环境收尾 134，simplex 长测排队；若三档全成功则自动进入 `minicpm_opt_p3_duplex` A/B/A |
| P3B shm-generation | `exp/p3b-shm-generation` | `/workspace/user_data/vllm-omni-worktrees/p3b-shm-generation` | 工作区候选（未提交） | 同一 req 的每次 SharedMemory transfer 使用 generation key；仅在 P3 header framing simplex 未通过时自动运行，避免混合变量 |
| P4 qsize-poll | `exp/p4-qsize-poll` | `/workspace/user_data/vllm-omni-worktrees/p4-qsize-poll` | `b401fe37` | StagePool 输出队列仅在非空时进入 wait；长测排队 |
| P5 prompt-scan | `exp/p5-prompt-scan` | `/workspace/user_data/vllm-omni-worktrees/p5-prompt-scan` | `de94dcdd` | talker prompt 长度扫描保持语义的单次遍历；长测排队 |
| P2 stage2-graph | `exp/p2-stage2-graph` | `/workspace/user_data/vllm-omni-worktrees/p2-stage2-graph` | `aebd7bc4` | Stage2 `enforce_eager=false` 对照；长测排队，需关注 graph 首轮开销与全双工 |
| P6-F stage2-mem | `exp/p6f-stage2-mem` | `/workspace/user_data/vllm-omni-worktrees/p6f-stage2-mem` | `4971f783` | Stage2 `gpu_memory_utilization=0.30` 容量对照；长测排队，不能与速度结论混淆 |
| P6-G prefix-cache | `exp/p6g-prefix-cache` | `/workspace/user_data/vllm-omni-worktrees/p6g-prefix-cache` | `ad97f501` | Stage2 `enable_prefix_caching=true`；长测排队，检查 cache 与跨 turn 状态隔离 |
| P6-H stage2-sync | `exp/p6h-stage2-sync` | `/workspace/user_data/vllm-omni-worktrees/p6h-stage2-sync` | `1d31d9ab` | Stage2 `async_scheduling=false` 对照；长测排队，需比较调度延迟与吞吐 |

截至本实验归档时，主工作区 `minicpm-challenge` 只保留默认配置（`n_timesteps=10`、float16 未启用）和实验文档。2026-08-17 按用户要求，P6-A 已通过合并提交 `8555eeb9` 合入主分支，当前默认 `token2wav_n_timesteps=3`；该合入明确保留已测得的 WER/SIM 回退风险，不应标记为通过 AGENT.md 无损精度门禁。

## 串行长测队列

为避免 NPU 并发占用，`minicpm_opt_queue_after_chunk20` 等 tmux watcher 会等待上一
个候选的 pytest 会话退出后再启动下一候选。已完成顺序为：chunk20 → chunk30 →
context0 → context6；随后 `minicpm_opt_queue_after_p6d` 完成 P1 A/B/A+Seed，
`minicpm_opt_queue_p6e` 执行 Stage 2 `max_num_seqs=8` 的 A/B/A+Seed，最后
`minicpm_opt_queue_p3` 执行 SHM framing 的 A/B/A+Seed。
若 P3 B 的三档 simplex 全部请求成功，`minicpm_opt_p3_duplex` 会继续执行默认/P3/默认
全双工 A/B/A；若 simplex 已有失败，则只记录跳过原因，避免将无效候选扩大到双工长测。
每一步均写入 `challenge_docs/run_log/` 和 `challenge_docs/batch_result/`，主分支不
切换配置。
