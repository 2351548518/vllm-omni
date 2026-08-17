# P6-H：Stage 2 `async_scheduling=false` 边界对照

## 候选

- worktree：`/workspace/user_data/vllm-omni-worktrees/p6h-stage2-sync`
- 分支/提交：`exp/p6h-stage2-sync` / `1d31d9ab`
- 唯一变量：仅 Stage 2 增加 `async_scheduling: false`；Stage 0/1、Stage 2 eager、
  sampling、chunk/context 和 full-duplex overlay 不变。
- 目的：确认 Stage2 async runner/deepcopy 是否实际带来收益或开销；该开关可能增加
  host blocking 和 TTFP，不能作为默认优化假设。

## 测试队列

等待 P6-G prefix-cache A/B/A+Seed 后，执行默认 A1 → P6-H B → 默认 A2 → P6-H
Seed-TTS。记录 scheduler mode、三档指标、SHM/PCM/失败统计和 Seed WER/SIM；任何
时序、性能或精度回退直接拒绝。

## 结果

待长测完成后填写；默认 Stage 2 async scheduling 保持现状。
