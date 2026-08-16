# 实验 worktree 索引

每个长测候选均保留独立分支和 worktree，避免后续恢复配置时覆盖实际测试代码。结果日志仍按实验名写入 `challenge_docs/run_log/` 和 `challenge_docs/batch_result/`。

| 实验 | 分支 | worktree | 固定提交 | 变量/状态 |
|---|---|---|---|---|
| P1 clone-view | `exp/p1-clone-view` | `/workspace/user_data/vllm-omni-worktrees/p1-clone-view` | `46b99f7d` | Code2Wav 两处 request cache clone→detached view；已完成单次性能/Seed，A/B/A/full-duplex 待补 |
| P6-A n3 | `exp/p6a-n3` | `/workspace/user_data/vllm-omni-worktrees/p6a-n3` | `b65a12e9` | `token2wav_n_timesteps=3`；性能改善但 Seed WER/SIM 回退，拒绝 |
| P6-B float16 | `exp/p6b-float16` | `/workspace/user_data/vllm-omni-worktrees/p6b-float16` | `c287518b` | `token2wav_float16=true`；等待 A 基线 Seed 完成后运行 |

主工作区 `minicpm-challenge` 当前只保留默认配置（`n_timesteps=10`、float16 未启用）和实验文档；主工作区 A 基线任务可安全运行，后续候选从对应 worktree 启动。
