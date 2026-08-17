# P5：Stage1 bridge 的 talker prompt 单次扫描

## 候选

- worktree：`/workspace/user_data/vllm-omni-worktrees/p5-prompt-scan`
- 分支/提交：`exp/p5-prompt-scan` / `de94dcdd`
- 唯一代码点：`compute_talker_prompt_ids_length` 从“先收集全部起点再二次遍历”改为
  单次扫描，保留 system/user/assistant 角色判定和最后 assistant 的 `+9` 语义。
  不改变 codec history、sampling、epoch/turn/fence 或 payload 字段。
- 动机：Stage1 bridge 每次条件/streaming chunk 可能重复解析 prompt token list；该补丁
  只减少 Python list 分配/遍历，优先验证无语义变化后再看端到端收益。

## 测试队列

等待 P4 qsize-poll 的 A/B/A+Seed 结束后，执行默认 A1 → P5 B → 默认 A2 → P5
Seed-TTS。原始日志/JSON 使用 `p5_*` 前缀。若 prompt 长度、请求失败、音频 PCM、
Seed WER/SIM 或 full-duplex metadata 有任一回退，保留分支并拒绝。

## 结果

待长测完成后填写三档指标和 Seed JSON；当前不修改默认 bridge。
静态预检：`adapter.py` 已通过 Python 3.12 `py_compile`；Chapter 10 性能与 Seed-TTS
仍按串行队列等待 P4 完成后执行。
