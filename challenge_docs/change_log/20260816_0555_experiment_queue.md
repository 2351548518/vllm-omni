# A3 推理优化连续实验队列

## 执行规则

- 每轮只改变一个参数或一个同质代码点；测试命令使用 Chapter 10 的 pytest 入口，由 fixture 自动启动服务。
- 每轮保存：候选 patch/config、完整 stdout/stderr、性能 JSON、Seed-TTS WER/SIM JSON、失败/PCM/时长统计和结论。
- 普通性能固定 `(1,32)/(4,64)/(8,128)`；精度先跑 Seed-TTS 全量 2020 条；候选只有在精度无回退后才进入下一轮。
- 性能和精度不得并发占用 NPU。每个参数候选按 `A-B-A` 排队；未完成的轮次不得标记为通过。

## 队列

| 顺序 | 轮次 | 实验 | 状态 | 备注 |
|---:|---|---|---|---|
| 1 | P1 | Code2Wav request-state redundant clone → detached view | 已拒绝 | A1/B/A2 在 8/128 均遇 SHM trailing bytes（完成约 25/25/26）；B Seed 仅 137/2020 完成、1883 失败、1 无 PCM，WER 2.6308%，单次 WER 1.3606% 不能作为通过证据 |
| 2 | P6-A | `token2wav_n_timesteps: 10 → 3 → 10` | 已完成，拒绝 | 性能显著提升，但 Seed-TTS WER 1.3875%→1.4542%、SIM 0.848603→0.846267；已恢复 10 |
| 3 | P6-B | `token2wav_float16: false → true → false` | 已完成，拒绝 | CFM `float` 输入与 `Half` bias dtype mismatch；性能/Seed 均失败，已恢复默认 false |
| 4 | P6-C | `codec_chunk_frames: 25 → 20/30` | chunk20/30 均拒绝；chunk30 触发 SHM MessagePack 崩溃 | chunk30 性能改善但约 395/2020 时 `MessagePack data is malformed: trailing characters (byte 1)`，服务退出；保留失败日志 |
| 5 | P6-D | `codec_left_context_frames: 3 → 0/6` | context0/context6 均拒绝 | context0 约 133/2020、context6 约 128/2020 触发同一 MessagePack trailing bytes；context6 Seed 完成 137/2020、1883 失败、1 无 PCM |
| 6 | P6-E | stage capacity (`max_num_seqs`, `max_num_batched_tokens`) | 已完成，拒绝 | A1/B/A2 在 8/128 分别完成 25/27/20，均复现 SHM；Seed 仅完成 136/2020、失败 1884、无 PCM 2（WER 1.4155%、SIM 0.848637 仅基于 134 条），已归档并拒绝 |
| 7 | P3 | SHM/ChunkTransfer payload length framing（`exp/p3-shm-length`） | 已完成，拒绝 | A1/B/A2 的 8/128 均 27/128 成功、101 失败；Seed 仅 138/2020 完成、1882 失败、无 PCM 2（WER 1.9119%、SIM 0.848210 仅基于 136 条），已拒绝并跳过全双工，转 P3B |
| 8 | P3B | SHM generation key（`exp/p3b-shm-generation`） | B 单工通过，A2/Seed 待完成 | clean 环境 A1/B 的三档均 100% 完成，越过旧 8/128 崩溃窗口；仍需默认 A2、全量 Seed-TTS 和 full-duplex，结果写入 `p3b_clean_*` |
| 9 | P4 | Orchestrator ready-queue / 1ms poll 优化 | 待启动 | 先状态机测试，再 A3 压测和 duplex |
| 10 | P5 | Stage1/bridge 小对象与 codec history 预分配 | 待启动 | 保留 epoch/turn/fence/terminal metadata |
| 11 | P2 | 稳定 Code2Wav bucket 的固定 shape graph/launch | 待启动 | 只试固定 shape，动态/终端走 fallback |
| 12 | P6-F | `gpu_memory_utilization` 容量实验 | 待启动 | 只当容量/排队实验，不能与其他变量混合 |
| 13 | P6-G | `enable_prefix_caching`、`async_chunk`、`async_scheduling` 等边界开关 | 待启动 | 仅做可回退对照，不默认保留风险开关 |

## 当前观察驱动的调整

P1 性能单次结果显示 Stage2 音频指标在低/中并发改善，高并发有小幅波动；Seed 全量 WER/SIM 尚未完成 A/B/A 与 full-duplex 验收。P6-A 的 n_timesteps=3 性能大幅改善，但全量 Seed-TTS WER/SIM 均回退，已拒绝并恢复默认 10。下一轮只测试 `token2wav_float16`，避免与 n_timesteps 混合。

## 2026-08-17 P6-A 用户指定合入

虽然 P6-A 按本队列的无损精度门禁应保持拒绝，用户基于性能收益与可接受的精度损失明确要求将其合入 `minicpm-challenge`。已通过合并提交 `8555eeb9` 合入，当前默认配置为 `token2wav_n_timesteps=3`；该状态不应被描述为精度门禁通过，full-duplex 仍未验证。

## 2026-08-16 队列编排纠错

P3B 启动后发现旧的 P4 watcher 只等待 `minicpm_opt_queue_p3`，没有等待条件候选
`minicpm_opt_p3b`，因此 P3B A1 与 P4 A1 曾短暂同时启动两个服务（端口
51359/58061），共享同一张 A3 NPU。该 P4 运行在进入请求前已安全中止，结果不纳入
任何结论；P5/P2/P6F/G/H 的等待 wrapper 也已停止，避免继续级联并发。

已将 `challenge_docs/result/run_p4_qsize_poll_queue_20260816.sh` 的前置条件改为等待
`minicpm_opt_p3b` 结束，再按 P4 → P5 → P2 → P6F → P6G → P6H 串行。此前已完成的
P3/P3B 结果仍按各自日志记录，后续性能数字不与这次并发启动混用。

随后检查到 `/dev/shm` 64 MiB tmpfs 已用满（约 34,561 个 `shm_*`、`chatcmpl-*` 和
`sem.*` benchmark 残留文件）；这些文件来自异常退出后的服务清理失败。P3B 在清理前
只完成了 1/32，未形成可用 A/B/A 结果，原始日志保留但标为环境污染。已停止所有等待
队列和服务，删除确认属于本轮 vLLM-Omni benchmark 的临时 SHM 文件，空间恢复为 0%
使用；P3B 改用 `p3b_clean_*` 独立目录从干净环境重跑。该清理解决的是 SHM 容量污染，
不能替代 generation-key 对同名段复用竞态的代码验证。
