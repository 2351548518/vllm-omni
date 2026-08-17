# P6-G：`enable_prefix_caching=true` 边界对照

## 候选

- worktree：`/workspace/user_data/vllm-omni-worktrees/p6g-prefix-cache`
- 分支/提交：`exp/p6g-prefix-cache` / `ad97f501`
- 唯一变量：deploy 顶层 `enable_prefix_caching: false → true`。不改采样、chunk、
  Token2Wav、stage capacity 或 duplex overlay。
- 风险：多媒体 prompt、异步 chunk、session/epoch/turn 状态的 cache key 可能不完整；
  即使性能改善，只要出现跨请求状态串扰、PCM/事件错序或精度回退即拒绝。

## 测试队列

等待 P6-F 容量实验 A/B/A+Seed 后，执行默认 A1 → P6-G B → 默认 A2 → P6-G
Seed-TTS。记录 prefix cache hit/miss、KV/HBM、三档指标、失败/无 PCM、Seed WER/SIM。
该实验只作可回退对照，不直接进入正式配置。

## 结果

待长测完成后填写。
