# vLLM-Omni MiniCPM-o 4.5 Infra 竞赛工作约束

本文档适用于本仓库及 `minicpm-challenge` 分支的后续开发。目标来自
`challenge_docs/Official_guide/challenge.md`：参加 vLLM-Omni 子赛道，在 Ascend
910C/NPU 上优化 MiniCPM-o 4.5 的推理性能。

## 目标优先级

1. 在官方评测环境中降低流式语音生成的 chunk RTF、TTFT 和 TTFP。
2. 保持 MiniCPM-o 4.5 的文本、图像、音频、视频理解和语音输出能力。
3. 保持全双工实时语音链路可用、稳定、可复现。
4. 只做能被测量验证的必要改动；不为重构而重构。

## 不可违反的约束

- 不得以牺牲精度换取性能。Daily-Omni、TTS-Seed、Video-MME 的结果必须与修改前可比；赛事准入的硬上限是相对官方基线下降不超过 2 个百分点，但本项目内部验收目标是无可测精度回退。
- 不得破坏半双工或全双工。全双工至少要继续支持 `/v1/realtime?duplex=1` 和 `/v1/duplex`，持续 PCM16 输入、listen/speak 决策、barge-in、音频/转写事件顺序、turn/epoch/fence 语义都必须保持。
- 不得改变评测语义来制造性能提升：不要随意修改 sampling、temperature、repetition penalty、模态 packing、帧数、输出长度、音频 chunk 大小或终止条件。若确需修改，必须说明等价性并重新做精度和 Demo 验证。
- 优先使用 NPU 专用、平台条件分支或局部实现；不得无理由改变 CUDA/ROCm/通用路径。
- 不升级比赛镜像中锁定的 torch、torch_npu、CANN 或 vLLM 运行时依赖。Token2Wav 依赖遵循官方文档：`stepaudio2-minicpmo` 与 `step-audio2 --no-deps`，避免覆盖 NPU 适配栈。
- 保护现有用户改动和未跟踪文件；不得使用破坏性 git 操作，不得擅自清理数据、日志或 benchmark 结果。

## 标准环境与启动方式

- 比赛目标硬件：单卡 Ascend 910C（A3）；官方镜像：`quay.io/ascend/vllm-omni:v0.25.0-a3`。
- 当前新增实测设备：Ascend A2/910B3。A2 与 A3 必须维护独立基线，只能在相同设备、镜像和测试配置内比较优化前后性能。
- 代码分支：`minicpm-challenge`。
- 多进程启动前设置：`export VLLM_WORKER_MULTIPROC_METHOD=spawn`。
- 普通三阶段服务使用 `vllm_omni/deploy/minicpmo_4_5.yaml`；需要拆卡或 TP 时使用对应 2/3 卡配置。
- 基准测试需记录实际镜像、vLLM/vLLM-Omni 安装来源和代码提交；当前环境的 `vLLM 0.1.dev1+...` 属版本字符串显示问题，与 vLLM-Omni 0.25.0 实际匹配，不得仅凭该告警否定环境或基线。
- 全双工必须使用 `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`，确认 `session_mode: duplex` 生效；`duplex_session.max_sessions` 必须与三个 stage 的 `max_num_seqs` 对齐。
- NPU 上优先走仓库内置的 MiniCPMO45 Token2Wav/in-tree `step_audio2_core` 后端，不要把带硬编码 `.cuda()` 的外部实现误接入 NPU 路径。

## 性能基线与指标

所有性能结论都必须记录硬件、镜像、代码提交、模型路径、deploy 配置、数据集版本、并发、warmup、测试次数和统计方式。不同设备的结果不得混为同一基线，也不得用 A2/910B3 与 A3/910C 的绝对差异证明代码优化。

当前 Seed-TTS 性能基线按 `(max_concurrency, num_prompts)` 的三组固定配对统计：`(1, 32)`、`(4, 64)`、`(8, 128)`。A3 数据来自 `challenge_docs/test_yaml/test_minicpmo_4_5.json`；A2/910B3 数据来自 `challenge_docs/batch_result/origin/simplex_performance/` 中 2026-08-14 的三份原始 JSON（结果字段 `Hardware` 显示为 `910B`）。

| 设备 | max concurrency | num prompts | request throughput | mean TTFT (ms) | mean E2EL (ms) | mean audio TTFP (ms) | mean audio RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| A3 / 910C | 1 | 32 | 0.5383 | 333.2633 | 1857.2154 | 986.4666 | 0.4423 |
| A3 / 910C | 4 | 64 | 0.6042 | 533.8656 | 6600.5101 | 3411.0754 | 1.5734 |
| A3 / 910C | 8 | 128 | 0.7547 | 547.3388 | 10450.6811 | 3352.7515 | 2.3024 |
| A2 / 910B3 | 1 | 32 | 0.2841 | 512.5854 | 3519.0786 | 1857.8724 | 0.8611 |
| A2 / 910B3 | 4 | 64 | 0.3299 | 551.0491 | 12053.6667 | 5747.8099 | 2.9701 |
| A2 / 910B3 | 8 | 128 | 0.3888 | 551.6746 | 20333.2862 | 7873.1024 | 5.0804 |

后续若重新跑出某设备的新基线，必须先把原始 JSON 保存到 `challenge_docs/batch_result/origin/`，再更新本表并记录日期、环境和来源。旧基线不得覆盖或删除。

重点比较 `audio_rtf`、`audio_ttfp`、`ttft` 和 `e2el`，同时检查吞吐、请求失败率、流式连续性、音频时长和显存/NPU 利用率。性能提升必须与同设备、同并发、同请求数的对应行比较。`challenge_docs/result/` 中的单项运行记录不能替代批量设备基线。

## 当前第一优先级优化假设：Token2Wav 3 steps

- 第一优先级 A/B 候选是把 `connectors.connector_of_shared_memory.extra.token2wav_n_timesteps` 从当前默认值 `10` 调整为 `3`。MiniCPM-o 4.5 的 Code2Wav 源码会读取该配置，多份非官方实测分享报告 TTFP/RTF 显著改善；这些分享只能作为实验线索，不能替代本机结果。
- 初次验证必须使用独立 deploy 覆盖配置，不要直接改写默认配置；只改变 `token2wav_n_timesteps` 一个变量，并按 `10 -> 3 -> 10` 的 A/B/A 顺序在同一设备、镜像、模型、数据和服务参数下测量。
- A2/910B3 至少覆盖正式三组 `(1, 32)`、`(4, 64)`、`(8, 128)`；A3/910C 到位后必须在 A3 上重新建立结论，不得把 A2 的相对收益直接当作 A3 结果。
- 降低步数会改变 Token2Wav 的流匹配/ODE 求解过程和最终波形，因此不得预设为“无损优化”。必须重新验证 Seed-TTS WER/SIM、无 PCM/失败计数、音频时长和听感/连续性；若 `3` 有任何可测质量回退，可测试 `5` 等中间值寻找无损性能点，但不得测试或保留无法正常初始化的配置。
- `minicpmo_4_5_duplex.yaml` 继承普通 MiniCPM-o 4.5 基础配置，所以该参数也会影响全双工音频输出。必须验证持续输入、listen/speak、barge-in、事件顺序、音频分块连续性、TTFP 和 RTF，不能只跑半双工性能。
- 只有三档性能结果可重复改善，精度无可测回退，且全双工门禁全部通过，才能把 `3`（或实测更优的无损中间值）写入正式配置并进入提交准备。

## 精度结果参考（910B 级资源）

以下是当前 910B 级资源的实测参考。`pytest` 门槛与赛事准入是不同口径：日常回归必须同时保存原始结果、成功/失败分母和评测配置；最终是否准入以当前官方文档和正式评测脚本为准，不得用代理指标或非官方分享替代官方口径。

| Benchmark | 实测 | pytest 门槛 | 赛事准入 |
| --- | --- | --- | --- |
| Seed-TTS WER | **3.3%**（1088/1088，ASR 失败 0） | ≤5% | WER ≤1.56（口径注意：vLLM-Omni 的 SIM 是 WavLM 代理，非官方 UniSpeech ASV） |
| Daily-Omni | **77.94%**（1197/1197） | ≥78%（差 0.06pp，pytest 会红） | ≥77.5%（基线 79.5，降 1.56pp） |
| Video-MME | **69.59%**（2700/2700，短 80.1/中 69.7/长 59.0） | ≥68% | ≥67.0% |

Daily-Omni 当前实测低于仓库 pytest 门槛 `0.06pp`，因此现状不能被描述为“pytest 全绿”；优化前后必须使用完全相同的数据、参数和成功请求分母比较。Seed-TTS 的 WavLM SIM 只能作为回归代理，若赛事使用 UniSpeech ASV，应另行执行或保留对应官方结果。

## 修改与测试流程


1. 先建立未修改基线：用同一镜像、模型、配置、数据和命令跑目标 benchmark，保存原始输出。
2. 用日志、profiling、NPU 算子/图信息或阶段时间拆分定位瓶颈，再提出最小补丁。优先检查 stage 调度、NPU graph/PIECEWISE、AR runner、Thinker→Talker bridge、SharedMemory、Talker codec sampling、Code2Wav/HiFT 和 CPU/NPU 交界。
3. 修改后先跑受影响的 CPU 单元测试和静态检查，再跑 NPU 单请求 smoke、普通多模态/语音 smoke 和全双工 Demo/协议测试。
4. 做精度回归：至少覆盖 Daily-Omni、TTS-Seed（WER/SIM 及无 PCM/失败计数）和 Video-MME；评测输入、packing、采样和成功请求分母必须与基线一致。
5. 做性能回归：同一测试配置重复测量，报告平均值、P50/P99（若脚本提供）、RTF、TTFT、TTFP、失败率和流式连续性。
6. 只有在精度与全双工均通过，且目标性能有可重复、可解释的提升并且没有关键指标回退时，才允许进入提交准备；否则保留结果，不提交代码。

## 测试与优化产物归档

所有测试、运行和优化记录必须写入以下固定目录，不得只留在终端、`/tmp` 或聊天记录中：

- 批量测试的性能与精度指标统一放在 `/workspace/user_data/vllm-omni/challenge_docs/batch_result/`。建议继续按 `accuracy/`、`simplex_performance/`、`duplex_performance/` 分目录，保留原始结果和汇总。
- 每次代码或配置修改的变更日志放在 `/workspace/user_data/vllm-omni/challenge_docs/change_log/`。日志须记录日期、基线提交/工作区、修改文件、修改原因、测试命令、结果、风险和回退方法。
- 每个优化思路必须写成独立 Markdown 文件，放在 `/workspace/user_data/vllm-omni/challenge_docs/idea_note/`。至少记录瓶颈证据、优化假设、影响链路、基线、计划测试命令、精度/全双工风险、验收条件、实测结果和最终状态，避免后续忘记测试方法。
- 单项测试结果放在 `/workspace/user_data/vllm-omni/challenge_docs/result/`，包括单个 benchmark、单测组合、smoke、协议验证或局部 profiling 的结果。
- 服务、benchmark、pytest、Demo 和 profiling 的完整 stdout/stderr 运行日志放在 `/workspace/user_data/vllm-omni/challenge_docs/run_log/`。
- 文件名应包含时间和主题，例如 `YYYYMMDD_HHMMSS_<topic>.md|json|log`；同一次优化的 idea、change、result、run log 应使用一致主题并互相引用。
- 不覆盖历史基线或失败结果。优化前后必须分别保存；失败、回退和无提升结果也要留档，以避免重复试验。
- 性能结论必须能从 `batch_result` 或 `result` 追溯到对应 `run_log`、测试命令、环境和代码状态；仅有口头结论不得进入提交判断。

## 必查测试入口

- 阶段桥接和流式 codec：
  `tests/model_executor/stage_input_processors/test_minicpmo_4_5_async_chunk.py`、
  `tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py`。
- 全双工控制与客户端：
  `tests/fullduplex/minicpmo45/`、
  `tests/entrypoints/test_async_omni_duplex.py`、
  `tests/e2e/online_serving/test_minicpmo_4_5_duplex.py`。
- 普通 MiniCPM 离线/在线和精度：
  `tests/e2e/offline_inference/test_minicpmo_4_5.py`、
  `tests/e2e/online_serving/test_minicpmo_4_5.py`、
  `tests/e2e/accuracy/minicpmo_4_5/test_minicpmo_4_5.py`。
- 编排错误、跨阶段路由和 Diffusion/通用调度：
  `tests/engine/test_orchestrator_error_handling.py`、
  `tests/engine/test_orchestrator_stage_input_bridge.py`、
  `tests/diffusion/test_diffusion_scheduler.py`、
  `tests/diffusion/test_diffusion_step_pipeline.py`。

## 提交门槛

- 不自动执行 `git add`、`git commit`、`git push` 或创建 PR。
- 代码提交只能发生在性能优化已通过上述精度、Demo、全双工和测试门槛之后，并且性能提升有基线/优化前后原始结果支撑。
- 若没有性能提升、提升不可重复、精度下降、全双工回归、测试失败或结果无法复现，应明确停止在工作区，不提交。
- 提交说明必须包含：瓶颈、最小改动、测试命令、硬件/软件环境、优化前后指标、精度结果、全双工验证结果和已知限制。

