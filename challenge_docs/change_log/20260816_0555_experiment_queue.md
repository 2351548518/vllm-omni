# A3 推理优化连续实验队列

## 执行规则

- 每轮只改变一个参数或一个同质代码点；测试命令使用 Chapter 10 的 pytest 入口，由 fixture 自动启动服务。
- 每轮保存：候选 patch/config、完整 stdout/stderr、性能 JSON、Seed-TTS WER/SIM JSON、失败/PCM/时长统计和结论。
- 普通性能固定 `(1,32)/(4,64)/(8,128)`；精度先跑 Seed-TTS 全量 2020 条；候选只有在精度无回退后才进入下一轮。
- 性能和精度不得并发占用 NPU。每个参数候选按 `A-B-A` 排队；未完成的轮次不得标记为通过。

## 队列

| 顺序 | 轮次 | 实验 | 状态 | 备注 |
|---:|---|---|---|---|
| 1 | P1 | Code2Wav request-state redundant clone → detached view | 单次性能/Seed 完成，候选通过 | WER 1.3606%、SIM 0.848419；A/B/A 与 duplex 待补 |
| 2 | P6-A | `token2wav_n_timesteps: 10 → 3 → 10` | 已完成，拒绝 | 性能显著提升，但 Seed-TTS WER 1.3875%→1.4542%、SIM 0.848603→0.846267；已恢复 10 |
| 3 | P6-B | `token2wav_float16: false → true → false` | A 性能运行中 | 已恢复 n_timesteps=10；先建立同配置 A，再测试 float16 |
| 4 | P6-C | `codec_chunk_frames: 25 → 20/30` | 待启动 | 先分别测 20、30；保留事件边界 |
| 5 | P6-D | `codec_left_context_frames: 3 → 0/6` | 待启动 | 连续性、边界音频和 duplex 是硬门禁 |
| 6 | P6-E | stage capacity (`max_num_seqs`, `max_num_batched_tokens`) | 待启动 | 一次只改一个 stage/变量，记录 HBM/KV/OOM |
| 7 | P3 | SHM/ChunkTransfer profiling 后的固定 header/ring 快速路径 | 待启动 | 先零语义 profiling，再最小实现 |
| 8 | P4 | Orchestrator ready-queue / 1ms poll 优化 | 待启动 | 先状态机测试，再 A3 压测和 duplex |
| 9 | P5 | Stage1/bridge 小对象与 codec history 预分配 | 待启动 | 保留 epoch/turn/fence/terminal metadata |
| 10 | P2 | 稳定 Code2Wav bucket 的固定 shape graph/launch | 待启动 | 只试固定 shape，动态/终端走 fallback |
| 11 | P6-F | `gpu_memory_utilization` 容量实验 | 待启动 | 只当容量/排队实验，不能与其他变量混合 |
| 12 | P6-G | `enable_prefix_caching`、`async_chunk`、`async_scheduling` 等边界开关 | 待启动 | 仅做可回退对照，不默认保留风险开关 |

## 当前观察驱动的调整

P1 性能单次结果显示 Stage2 音频指标在低/中并发改善，高并发有小幅波动；Seed 全量 WER/SIM 尚未完成 A/B/A 与 full-duplex 验收。P6-A 的 n_timesteps=3 性能大幅改善，但全量 Seed-TTS WER/SIM 均回退，已拒绝并恢复默认 10。下一轮只测试 `token2wav_float16`，避免与 n_timesteps 混合。
