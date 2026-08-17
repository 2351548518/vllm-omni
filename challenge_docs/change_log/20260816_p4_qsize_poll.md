# P4：Orchestrator 空闲 poll 的输出队列快速检查

## 候选

- worktree：`/workspace/user_data/vllm-omni-worktrees/p4-qsize-poll`
- 分支/提交：`exp/p4-qsize-poll` / `b401fe37`
- 唯一代码点：`StagePool.poll_llm_raw_output` 在创建 `asyncio.wait_for` 前，对
  `AsyncMPClient.outputs_queue.empty()` 做 advisory 空闲检查；队列不可见/不支持
  `empty()` 时完全保留原超时路径。该补丁不改 scheduler 状态、1ms 轮询周期、输出顺序、
  duplex fence/turn 或请求终止语义。
- 动机：`Orchestrator._orchestration_loop` 对每个 stage/replica 每轮创建并取消
  1ms `wait_for`；高并发和空闲混合时存在 Python task/context-switch 开销。快速检查
  只跳过确定为空的 queue，下一轮仍会重新观察。

## 测试队列

等待 P3 SHM framing 的 A/B/A+Seed 退出后，执行默认 A1 → P4 B → 默认 A2 → P4
Seed-TTS。原始日志/JSON 按 `p4_*` 保存；若出现 output queue race、请求失败、PCM
断裂、性能回退或精度回退，立即拒绝并保留分支。

## 结果

待 A/B/A+Seed 长测完成后填写。full-duplex 只有在 simplex 性能与 Seed 均通过后才会
进入验证，当前不改默认 orchestrator。
静态预检：`stage_pool.py` 已通过 Python 3.12 `py_compile`；Chapter 10 性能与 Seed-TTS
仍按串行队列等待 P3 完成后执行。
