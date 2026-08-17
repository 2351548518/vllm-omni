# P2：Stage 2 Code2Wav graph/compile 对照实验

## 候选

- worktree：`/workspace/user_data/vllm-omni-worktrees/p2-stage2-graph`
- 分支/提交：`exp/p2-stage2-graph` / `aebd7bc4`
- 唯一变量：Stage 2 `enforce_eager: true → false`；Stage 0/1、Code2Wav steps、
  chunk/context、sampling、bridge 和 duplex 配置不变。
- 动机：Stage 2 当前显式 eager，若固定 bucket 可被 NPU compile/graph 复用，可能减少
  host launch；但 Code2Wav 含动态 batch/state/终端边界，预计存在 graph miss 或启动失败。
  该候选只用于实测，不预设可行。

## 测试队列

等待 P5 prompt-scan 的 A/B/A+Seed 结束后，执行默认 A1 → P2 B → 默认 A2 → P2
Seed-TTS。必须记录 compile/cudagraph hit/miss、启动/OOM、失败请求、PCM 连续性和
Seed WER/SIM；任何初始化失败或动态边界语义问题即拒绝并保留分支。

## 结果

待长测完成后填写；默认 Stage 2 eager 在门禁前保持不变。
静态预检：候选 deploy YAML 已通过 `yaml.safe_load`；Chapter 10 性能与 Seed-TTS
仍按串行队列等待 P5 完成后执行。
