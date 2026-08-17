# P3：SharedMemory payload length framing

## 候选

- 独立 worktree：`/workspace/user_data/vllm-omni-worktrees/p3-shm-length`
- 分支/提交：`exp/p3-shm-length` / `ae707303`
- 代码：`shm_write_bytes` 在 payload 前写固定 magic+长度 header；`shm_read_bytes` 优先
  按 header 截取，旧无 header segment 走原 `meta["size"]` fallback；SHM connector 返回
  实际逻辑 payload 长度而非 backing allocation 大小。
- 动机：chunk30、context0、context6 均在高并发/长 Seed-TTS 复现
  `MessagePack data is malformed: trailing characters (byte 1)`，核心怀疑是 key-only
  receiver 使用 backing `shm.size` 而不是 sender logical length。该 patch 不合并 chunk、
  不改事件/turn/fence、采样或数据语义。

## 静态/单测

`python -m compileall` 通过；`tests/distributed/omni_connectors/test_shm_connector.py`
的 16 个断言全部通过，但进程收尾出现环境已知的 `corrupted size vs. prev_size`，pytest
退出码 134。该现象已记录原始输出，不能单独视为 NPU 运行通过；需结合长测和现有基线
对比。

## 长测队列

等待 P6-E 的 A/B/A+Seed 队列退出后，串行执行默认 A1 → P3 B → 默认 A2 → P3 Seed-TTS。
原始 stdout/stderr 位于 `challenge_docs/run_log/p3_*`，性能 JSON 位于
`challenge_docs/batch_result/p3_shm_length_*`，不覆盖 baseline。重点观察 SHM 错误、
失败/无 PCM、音频时长和 8/128 稳定性；任一错误或精度回退即拒绝。

## 结果

待长测完成后填写 A/B/A 指标、Seed WER/SIM 和退出码。未通过稳定性、精度及全双工门禁
前，不修改默认 `minicpmo_4_5.yaml`。

### 默认 A1 性能（已完成，基线仍失败）

日志：`challenge_docs/run_log/p3_a1_default_simplex_20260816_120000.log`。

| 并发/请求数 | 完成/失败 | 吞吐 (req/s) | TTFT (ms) | E2EL (ms) | audio TTFP (ms) | audio RTF |
|---|---:|---:|---:|---:|---:|---:|
| 1/32 | 32/0 | 0.494895 | 316.670 | 2020.243 | 1062.784 | 0.497845 |
| 4/64 | 64/0 | 0.555624 | 459.799 | 7130.871 | 3591.731 | 1.769461 |
| 8/128 | 27/101 | 0.767473 | 464.247 | 9356.783 | 3866.148 | 2.641583 |

默认 A1 的 8/128 在约第 27 条再次报 `SharedMemoryConnector ... MessagePack data is
malformed: trailing characters (byte 1)`，随后服务退出；pytest 为 1 failed。该 A1 先
确认当前基线故障仍可复现，P3 B 仍按队列执行以判断 framing 是否改变故障阈值。

### P3 B framing 性能（已完成，拒绝）

日志：`challenge_docs/run_log/p3_b_shm_length_simplex_20260816_120000.log`。

| 并发/请求数 | 完成/失败 | 吞吐 (req/s) | TTFT (ms) | E2EL (ms) | audio TTFP (ms) | audio RTF |
|---|---:|---:|---:|---:|---:|---:|
| 1/32 | 32/0 | 0.516469 | 316.740 | 1935.823 | 1034.878 | 0.475570 |
| 4/64 | 64/0 | 0.587205 | 474.307 | 6756.134 | 3433.267 | 1.683595 |
| 8/128 | 27/101 | 0.809829 | 486.326 | 8753.607 | 3499.556 | 2.550250 |

8/128 在约第 27 条仍报 `MessagePack data is malformed: trailing characters (byte 1)`，
与默认 A1 的失败阈值一致，pytest 失败。header+logical-length framing 没有改变 key
生命周期竞争；即使低/中并发局部指标较好，也不能接受。A2 与 Seed 继续执行仅为形成
完整对照并决定后续 P3B generation-key 队列。
### 默认 A2 性能（已完成，基线仍失败）

日志：`challenge_docs/run_log/p3_a2_default_simplex_20260816_120000.log`。

| 并发/请求数 | 完成/失败 | 吞吐 (req/s) | TTFT (ms) | E2EL (ms) | audio TTFP (ms) | audio RTF |
|---|---:|---:|---:|---:|---:|---:|
| 1/32 | 32/0 | 0.520548 | 317.455 | 1920.702 | 1027.052 | 0.472172 |
| 4/64 | 64/0 | 0.599059 | 478.881 | 6590.336 | 3407.704 | 1.639719 |
| 8/128 | 27/101 | 0.811872 | 498.879 | 8785.695 | 3503.618 | 2.539385 |

默认 A2 的 8/128 再次在第约 27 条触发同一 trailing-bytes，pytest 失败。A1、P3 B、A2
三组性能现场均已完成高并发失败归档；Seed-TTS 仍按队列执行。

### P3 Seed-TTS（已完成，拒绝）

日志：`challenge_docs/run_log/p3_shm_length_seed_accuracy_20260816_120000.log`；结果：
`challenge_docs/batch_result/p3_shm_length_accuracy/qwen_omni_acc_seed_tts_20260816-140856.json`。

固定 2020 条、并发 4 的请求在约 155 条触发同一 `MessagePack data is malformed: trailing
characters (byte 1)`，最终完成 `138/2020`、失败 `1882`、无 PCM `2`。仅 `136` 条进入
WER/SIM，WER 均值 `1.9119%`、SIM 均值 `0.848210`；pytest 的已完成样本阈值通过不能
覆盖请求失败。P3 framing 因性能 8/128 与长 Seed 均失败，正式拒绝，不进入全双工。
