# P3B：SharedMemory transfer generation key（条件实验）

## 状态

独立 worktree：`/workspace/user_data/vllm-omni-worktrees/p3b-shm-generation`；分支：
`exp/p3b-shm-generation`。当前工作区候选尚未提交，等待 P3 payload-length framing
实验结束后由 `minicpm_opt_p3b` 条件 watcher 决定是否运行。

## 变量与动机

P3B 只改变 SharedMemoryConnector 的 transfer key 生命周期：`try_send_via_connector`
对每次 SHM transfer 使用 `req_id_<uuid>`，通知增加 `connector_key`，接收端优先使用该
key；原始 `request_id` 保持不变，继续用于 orchestrator 路由、指标和 turn 语义。非 SHM
connector 仍使用原 request_id key。这样可以避免同一个流式请求多次发送时 unlink/recreate
同名 POSIX segment 与旧 reader handle 并存。

该实验不改变 chunk/context、sampling、dtype、scheduler 或 payload 字段；P3 header
framing 若已通过则本轮自动跳过，避免把两个 SHM 假设混为一个结论。

## 预检

- `adapter.py` 与 adapter flow test 已通过 Python 3.12 `py_compile`。
- `test_shm_connector_flow` 已增加 generation key 断言；完整 pytest 等 NPU 队列空闲后运行，
  避免与长测共享运行时资源。

## 长测命令

条件 watcher：`challenge_docs/result/run_p3b_shm_generation_queue_20260816.sh`。若 P3
simplex 未通过，自动使用 Chapter 10 的普通性能 pytest A/B/A，再运行 Seed-TTS WER/SIM
全量 2020 条；日志与 JSON 分别写入 `challenge_docs/run_log/`、`challenge_docs/batch_result/`。

## 验收

必须三档请求全成功、Seed-TTS 无失败/无 PCM 且 WER/SIM 不回退；若稳定性修复有效但性能
无提升，只作为修复 SHM correctness 的候选，不以吞吐单点改善替代 full-duplex 门禁。

## 复测卫生与 full-duplex 门禁（2026-08-16）

首次 watcher 启动时发现 P4 旧队列未等待 P3B，并且 `/dev/shm` 64 MiB tmpfs 已被约
34,561 个异常退出残留的 benchmark 文件占满；该次 P3B 只到服务初始化/1 个低并发
结果，原始日志保留但不计入 A/B/A。已停止并发队列，清理确认属于 benchmark 的
`shm_*`、`chatcmpl-*`、`sem.*` 文件，并在每个性能 leg 收尾加入等待服务退出后的同样
清理。干净复测改用 `p3b_clean_*` 结果目录。

新增 `run_p3b_duplex_ab_a_queue_20260816.sh`：只有 P3B 单工三档全成功且 Seed-TTS
全量门禁通过时才运行默认 A1 → P3B B → 默认 A2 full-duplex；P4 队列等待该 watcher
结束后再启动，确保候选不会绕过双工验收或与下一轮共享 NPU。

## 干净复测 A1（进行中）

日志：`challenge_docs/run_log/p3b_clean_a1_default_simplex_20260816_143000.log`。
截至目前三档均完整成功，且服务没有复现旧的 trailing-bytes：

| 并发/请求数 | 完成/失败 | 吞吐 (req/s) | TTFT (ms) | E2EL (ms) | audio TTFP (ms) | audio RTF |
|---|---:|---:|---:|---:|---:|---:|
| 1/32 | 32/0 | 0.511524 | 316.775 | 1954.598 | 1031.591 | 0.481292 |
| 4/64 | 64/0 | 0.571362 | 467.442 | 6911.600 | 3522.705 | 1.716068 |
| 8/128 | 128/0 | 0.720415 | 458.842 | 10985.152 | 4352.489 | 2.736346 |

8/128 已越过此前默认/P1/P3/P6-E 在约 20–27 条请求触发 SHM 崩溃的窗口；A1 仍需与
generation-key B、默认 A2、全量 Seed-TTS 和 full-duplex 一起完成门禁，不能仅凭这一
组性能结果接受。

## 干净复测 A2 默认（已完成）

日志：`challenge_docs/run_log/p3b_clean_a2_default_simplex_20260816_143000.log`。
默认代码在 A2 三档同样全量成功，未出现 SHM 协议错误：

| 并发/请求数 | 完成/失败 | 吞吐 (req/s) | TTFT (ms) | E2EL (ms) | audio TTFP (ms) | audio RTF |
|---|---:|---:|---:|---:|---:|---:|
| 1/32 | 32/0 | 0.512400 | 314.330 | 1951.219 | 1028.746 | 0.480670 |
| 4/64 | 64/0 | 0.530491 | 472.264 | 7473.830 | 3792.344 | 1.842919 |
| 8/128 | 128/0 | 0.750543 | 463.346 | 10542.719 | 4206.662 | 2.629022 |

A1/B/A2 的请求稳定性门禁已通过（9 组均无失败）；仍须等待 Seed-TTS 全量精度和
full-duplex，才能判断 generation-key 是否可接受。

## 干净复测 B generation-key（已完成）

日志：`challenge_docs/run_log/p3b_clean_b_generation_simplex_20260816_143000.log`。
三档请求均全量完成，未出现 trailing-bytes、connection refused 或音频连续性错误：

| 并发/请求数 | 完成/失败 | 吞吐 (req/s) | TTFT (ms) | E2EL (ms) | audio TTFP (ms) | audio RTF |
|---|---:|---:|---:|---:|---:|---:|
| 1/32 | 32/0 | 0.517413 | 320.532 | 1932.223 | 1029.316 | 0.474561 |
| 4/64 | 64/0 | 0.597786 | 481.076 | 6614.028 | 3391.598 | 1.660835 |
| 8/128 | 128/0 | 0.707022 | 458.784 | 11190.308 | 4420.240 | 2.774212 |

B 的 8/128 也完整通过；下一步是默认 A2 复测与全量 Seed-TTS，若两者仍通过才进入
full-duplex watcher。
