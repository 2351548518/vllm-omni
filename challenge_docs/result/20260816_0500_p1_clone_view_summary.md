# P1 — Code2Wav 状态缓存 view 优化：结果摘要

## 范围

- 设备：A3 / Ascend 910C；模型：`/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5`。
- 变更：`vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py` 中移除两处不必要的 request-level `clone()`，保留 `detach()`；下一次 `_stack_flow_cache`/`torch.cat` 仍创建新的 batch tensor。
- 基线：`AGENT.md` 中 A3/910C 的 2026-08-15 同机重测结果。性能压测 JSON 自带的 `baseline.A3` 是更早的一轮，数值不同，以下不把它当作验收基线。

第十章命令核对：本轮只执行 pytest 入口，服务由性能 fixture / 精度 `omni_server` fixture 自动启动，随后内部调用 `vllm bench serve`；没有把手工常驻服务混入统计。Seed 日志已明确记录 `Launching OmniServer` → `Server ready` → `$ ... vllm bench serve ...`。

## 性能原始结果

日志：[`p1_clone_view_simplex_20260816_044425.log`](../run_log/p1_clone_view_simplex_20260816_044425.log)

| max concurrency / prompts | req/s | TTFT (ms) | E2EL (ms) | audio TTFP (ms) | audio RTF | completed/failed |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 32 | 0.514381 | 316.509 | 1943.761 | 1029.138 | 0.475683 | 32 / 0 |
| 4 / 64 | 0.607972 | 465.076 | 6491.151 | 3341.355 | 1.624081 | 64 / 0 |
| 8 / 128 | 0.743317 | 459.247 | 10669.581 | 4182.064 | 2.647707 | 128 / 0 |

原始 JSON：[`p1_clone_view_simplex`](../batch_result/p1_clone_view_simplex/)。pytest 结果为 `1 passed, 1 deselected, 15 warnings`，总耗时 `756.74s`。

## 对 AGENT.md A3 基线的相对变化

| 组 | 吞吐 | TTFT | E2EL | audio TTFP | audio RTF |
|---|---:|---:|---:|---:|---:|
| 1 / 32 | +3.85% | -0.46% | -3.70% | -2.22% | -3.59% |
| 4 / 64 | +5.83% | -9.99% | -5.34% | -7.23% | -5.48% |
| 8 / 128 | -0.94% | -10.82% | +1.09% | -3.15% | +0.88% |

这是单次候选运行，不足以通过 AGENT.md 要求的重复 A/B/A；高并发组存在小幅回退，需后续重复确认。

## 精度状态

Seed-TTS 过滤测试已在 tmux `minicpm_opt_p1:seed` 中启动，命令为 Chapter 10 的 `seed_tts_wer_bench and not duplex`，全量 2020 条、并发 4，同时开启 WER/SIM。原始日志为 [`p1_clone_view_seed_accuracy_20260816_045807.log`](../run_log/p1_clone_view_seed_accuracy_20260816_045807.log)，结果为 [`qwen_omni_acc_seed_tts_20260816-050235.json`](../batch_result/p1_clone_view_accuracy/qwen_omni_acc_seed_tts_20260816-050235.json)。

| 指标 | A3 baseline | P1 candidate | 变化 |
|---|---:|---:|---:|
| WER | 1.3875% | 1.3606% | -0.027pp |
| WavLM SIM | 0.848603 | 0.848419 | -0.000184 |
| completed / failed | 2020 / 0 | 2020 / 0 | 无变化 |
| no PCM / ASR failed / SIM failed | 0 / 0 / 0 | 0 / 0 / 0 | 无变化 |

性能与精度均为单次候选运行；full-duplex 尚未运行，A/B/A 仍待补齐。

## 判定

性能结果和日志已归档，但当前补丁保持“待验收”状态。只有 Seed-TTS 精度无回退、后续 A/B/A 性能重复通过并补做 full-duplex 门禁后，才可将其标为保留；否则恢复 `.clone()`。
