# MiniCPM-o 4.5：从 vLLM-Omni 核心链路到参数的推理加速思路

**日期**：2026-08-16  
**设备**：A3 / Ascend 910C  
**分支**：`minicpm-challenge`  
**状态**：`running / 已开始按优先级做长时 A/B/A；队列与结果见 change_log/20260816_0555_experiment_queue.md`
**范围**：普通流式、半双工、native full-duplex；目标是降低 audio TTFP、chunk RTF、TTFT 和 E2EL，不改变输出语义。

## 1. 约束与当前基线

本方案遵循 [`AGENT.md`](../../AGENT.md)：不改采样、packing、输出长度、终止条件和全双工协议语义；不升级锁定的 NPU 运行时；先做 profiling，再做单变量 A/B/A。任何参数或代码改动都必须同时通过精度、音频连续性和全双工门禁。

当前 A3 基线来自：

- 性能日志：[`baseline_simplex_performance_20260815_111642.log`](../run_log/baseline_simplex_performance_20260815_111642.log)，原始 JSON 在 `challenge_docs/batch_result/origin/simplex_performance/`。
- 精度日志：[`baseline_accuracy_20260815_113536.log`](../run_log/baseline_accuracy_20260815_113536.log)，原始 JSON 在 `challenge_docs/batch_result/origin/accuracy/`。

| 并发/请求数 | 吞吐 req/s | TTFT ms | audio TTFP ms | E2EL ms | audio RTF |
|---|---:|---:|---:|---:|---:|
| 1 / 32 | 0.4953 | 317.99 | 1052.49 | 2018.44 | 0.4934 |
| 4 / 64 | 0.5745 | 516.68 | 3601.83 | 6857.24 | 1.7182 |
| 8 / 128 | 0.7504 | 514.98 | 4318.23 | 10554.92 | 2.6245 |

质量基线为 Seed-TTS WER **1.39%**、WavLM SIM **0.8486**，Daily-Omni **78.03%**（934/1197，解析失败 2），Video-MME **69.59%**；native duplex 辅助测量为 200 turns、WER **1.93%**、SIM **0.8438**、无失败/无 PCM、连续性 100%、TTFP **1305.29 ms**、RTF **0.5177**。

## 2. 推理链路与已定位的热点

请求路径是：

```text
AsyncOmniEngine
  -> Orchestrator
  -> StagePool / StageEngineCoreClient
  -> vLLM Scheduler + NPU ModelRunner
  -> Stage 0 Thinker
  -> Stage 1 Talker / codec token
  -> SharedMemoryConnector / ChunkTransferAdapter
  -> Stage 2 Code2Wav / HiFT
  -> PCM16 + full-duplex event/fence/epoch/turn
```

### 2.1 Stage 2 是第一主热点：Code2Wav CFM/HiFT

代码证据：

- `vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py:229-286` 的 `_decode_cfm` 每个 chunk 默认执行 `n_timesteps` 次；每个 step 都重新 `torch.cat((x,x))`、构建时间/CFG 输入，并通过 `_estimator_buffers` 创建新的 CNN/attention cache。
- 同文件 `:288-326` 会在 batch 与每个 request 状态之间反复 split/stack/cat/clone；`:392-478` 还会拼接 HiFT 历史 mel/source/speech、做 overlap fade，并逐 request `detach().clone()` 和转 float32。
- `minicpmo_4_5_code2wav.py:457-472` 的 bucket key 包含 prompt、token 数、完整 state shape、终止状态和 epoch，保证状态隔离但也造成大量小而不相同的 batch。
- A3 日志的阶段耗时样本显示 Stage2 经常约 **7.9–15.1 s**，而同一请求 Stage0/Stage1 多为约 **0.4–3.2 s**；并发 4/8 时 E2EL、TTFP、RTF 随 Stage2 排队明显放大。

**核心代码优化假设（P0/P1）**：在不改变 CFM 数值递推顺序和状态边界的前提下，为固定 shape bucket 预分配并复用 estimator cache、时间向量、CFG 拼接缓冲和 HiFT 工作区，减少 NPU allocator、Python loop、cat/clone/split；然后再评估 NPU 固定 shape graph/capture。不能把“减少步数”冒充代码优化：步数是独立参数实验，必须单独归因。

### 2.2 Stage 1 与 bridge 是次热点：CPU 组装和跨进程复制

- `minicpmo_4_5_omni_tts.py:247-363` 在首个 prefill/每次条件变化时重新构造 condition embeddings；跨阶段传输的 CPU list 会再 `torch.as_tensor`。
- `:373-406` 每个 codec token 都会做 logits float 转换、repetition penalty、top-k/top-p、softmax、`multinomial`；这应先测量 CPU/NPU 占比，不能直接改变采样。
- `:408-569` 用 Python 循环逐 request 构建 stop logits、codec delta、duplex epoch/turn/text 元数据，并且 codec history 反复 `torch.cat`。
- `stage_input_processors/minicpmo_4_5_omni.py:206-385` 每个 async chunk 维护 Python pending list，把 tensor 转 list，再新建 `torch.tensor` 和共享内存 payload。chunk boundary、`segment_end`、`turn_end` 等是全双工协议的一部分，不能靠减少复制而丢失。

**核心代码优化假设（P3）**：缓存 token id/boundary 查找和稳定的小张量；对固定窗口预留 codec history，避免每 token cat；普通 simplex 先做只读/无 alias 的快速路径，native duplex 复用同一数据结构但保留现有字段和 fence 检查。隐状态、音频 codes 不能引用会被 runner/detokenizer 后续修改的 buffer。

### 2.3 Connector/SHM 是跨阶段边界热点

- `distributed/omni_connectors/connectors/shm_connector.py:37-65` 每个 chunk 做一次序列化、打开 `/dev/shm` lock 文件、独占锁写入；`:67-143` 再独占锁读、反序列化并删除 lock 文件。
- `distributed/omni_connectors/adapter.py:48-89` 把 engine inputs、sampling params 和 original prompt 组装成字典后再发轻量通知；接收端 `:131-152` 解码并检查 payload。
- `transfer_adapter/base.py:49-83` 的接收线程按 pending request 轮询，失败时以 1ms 条件等待；空闲时 100ms 唤醒。`chunk_transfer_adapter.py` 还会在 scheduler tick 中反复 park/restore waiting/running 队列。

**核心代码优化假设（P2）**：A3 同机 SharedMemory 路径增加一次性 payload header/固定 tensor view 或预分配 ring，合并同一唤醒周期内的 ready chunks，避免 `.tolist()`、重复 dict merge、flatten/unflatten 和重复 SHM open；仅在 `SharedMemoryConnector` 且同机时启用，其他 connector 走原路径。必须保留 request 顺序、chunk_seq、`segment_end`/`turn_end`、cache epoch、duplex epoch/turn/fence 和 terminal sentinel。

### 2.4 Scheduler/Orchestrator 热路径

- `engine/orchestrator.py:892-1037` 每轮遍历所有 stage 和 replica；LLM 以 `asyncio.wait_for(..., timeout=0.001)` 轮询，空闲后 sleep 1ms。并发 8 时会产生大量无输出的 poll/上下文切换。
- `:1691-2002` 每次跨阶段路由都要重新解析输入、检查 stage 状态、构造 `EngineCoreRequest`，并记录大量 per-request 字典字段。
- `core/sched/omni_ar_scheduler.py` 的 `schedule`/`update_from_output` 会逐 request 包装 `OmniNewRequestData`、查询 additional information、处理 connector signal、状态和 finish；源码已注明 `num_scheduled_tokens` 较大时逐 request loop 可能是瓶颈。
- `platforms/npu/worker/npu_ar_model_runner.py:430-452` 在 async/spec 或 PCP+MM 条件下会 `deepcopy(scheduler_output)`，源码明确标记为昂贵待优化。
- `worker/omni_connector_model_runner_mixin.py` 的 pending id snapshot、锁内 shallow-copy、finished set drain 会在每个 runner step 参与跨阶段同步。

**核心代码优化假设（P4）**：增加 ready queue/事件唤醒，只有有输出或 connector signal 时处理 replica；预计算 request id→index 和不变的 stage metadata；无 signal 时跳过 `OmniNewRequestData` 的重复包装；将 scheduler output 的 copy 限制在确实需要的 NPU async/spec 分支。不得改变 `WAITING_FOR_INPUT`、`WAITING_FOR_CHUNK`、RUNNING、KV transfer、abort 清理和 full-duplex stage0 resumable 语义。

### 2.5 NPU runner、KV 和 graph

Stage0/1 仍由 vLLM NPU AR runner 管理 KV block、prefill/decode 和 PIECEWISE graph；Stage2 在 MiniCPM 部署中显式 `enforce_eager: true`，且由 in-tree `MiniCPMO45Token2wav`（`minicpmo_4_5_code2wav.py:738-775`）承载自定义后端。优化方向应是固定 shape、减少 host sync 和 metadata/copy，而不是未经验证地将 Stage2 切换为通用 vLLM graph。

## 3. 参数变量审计（必须单变量，不直接改默认值）

| 参数 | 当前值/位置 | 影响链路 | 建议与风险 |
|---|---|---|---|
| `token2wav_n_timesteps` | 默认 10，`code2wav.py:774` 从 connector extra 读取 | Stage2 CFM loop 次数，直接影响 TTFP/RTF 和波形 | AGENT 指定的第一优先级 A/B/A：`10→3→10`；3 会改变 ODE 误差，必须做 WER/SIM、时长、听感和 duplex 回归，不能预设无损。可在 5 等中间值寻找无损点。 |
| `token2wav_float16` | 默认 false，`code2wav.py:764-775` | Token2Wav/HiFT dtype、NPU 带宽和精度 | 只在 NPU overlay 做 `false→true→false`；逐样本比较音频时长、PCM 有效性、WER/SIM。不能与 n_timesteps 同一轮改。 |
| `codec_chunk_frames` | A3 默认 25 | bridge 触发频率、Code2Wav batch shape、TTFP/RTF、事件数量 | 25→较小值可能降低首包等待但增加 scheduler/SHM/HiFT 次数；较大值可能提升吞吐但恶化 TTFP。必须保持 token/音频总长度和 turn/event 顺序，先做 `20/25/30` 独立实验。 |
| `codec_left_context_frames` | A3 默认 3 | encoder/HiFT 上下文、边界伪影、cache shape | 不能仅看速度调低；测试 `0/3/6` 时做 PCM 连续性、边界听感和 duplex barge-in。不要把其他模型的 25/72 直接套到 MiniCPM。 |
| `max_num_seqs` | Stage0/1/2 普通部署均 4；duplex 均 2 | scheduler admission、batch 大小、显存/HBM、full-duplex session 上限 | 并发 8 的 Stage2 可能排队，但增大有 OOM 和 shape bucket 碎片风险；普通与 duplex 分开测，duplex 必须与 `duplex_session.max_sessions` 对齐。 |
| `max_num_batched_tokens` | Stage0 16384、Stage1 8192、Stage2 65536；NPU Stage0/1 override 8192 | prefill/decode 合批上限和 KV 峰值 | 需结合 scheduler trace 看是否真的达到上限；只改一个 stage，观察 TTFT/TTFP、KV block、OOM。对 Stage2 不能以 token cap 推断 Code2Wav batch，因为 bucket 还受 state shape 约束。 |
| `gpu_memory_utilization` | 0.55 / 0.15 / 0.18 | KV capacity 与三个 stage 的资源竞争 | 只作为容量/排队实验，不是单纯“越高越快”；记录 HBM、KV preemption、OOM 和 audio RTF。 |
| `enforce_eager` / `compilation_config.cudagraph_mode` | Stage0/1 NPU PIECEWISE；Stage2 eager | graph capture、host launch、动态 shape 兼容 | Stage0/1 先验证固定 shape graph hit/miss；Stage2 只有在自定义 Token2Wav 支持等价 capture 后才试，不能直接把 eager 改 false。 |
| `enable_prefix_caching` | false | prompt/KV 复用 | 多媒体与可变 duplex turn 的 prompt/epoch 不稳定，存在错误复用风险；除非证明 cache key 覆盖 session/epoch/state，否则不启用默认值。 |
| `async_chunk` | true | Thinker→Talker→Code2Wav 重叠、首包和 full-duplex | 不得为吞吐关闭；应优化其 ready/queue/SHM 实现。同步模式仅作为故障定位对照。 |
| `async_scheduling` | 普通 AR 默认 true；duplex overlay 为 false | NPU runner 异步采样、状态修正、full-duplex 时序 | 不把切到 false 当优化；仅通过 runner profiling 判断 deepcopy/async copy 是否可局部消除。duplex 的 false 是现有稳定性边界。 |
| `active_stream_window` | duplex 为 1 | 同时推进的 stream 数和公平性 | 增大可能提高吞吐但改变实时音频调度/turn 顺序；需单独压力测试，不作为第一批改动。 |
| `connector_get_sleep_s`、`connector_get_max_wait*` | deploy 中 0.01 / 3000 / 300 | connector 配置表面参数 | MiniCPM 当前消费路径主要由 `OmniTransferAdapterBase` 的 1ms/100ms 条件轮询和 runner 注册逻辑控制，不能假设改 YAML 有效；先加计时/命中统计，确认消费端后再改。 |
| sampling（`temperature/top_p/top_k/min_tokens/max_tokens/repetition_penalty/seed`） | AGENT 记录的固定值 | 直接改变文本/codec 采样和精度 | 不纳入性能旋钮；任何改变都必须视为精度实验，不能与引擎优化混合。 |

## 4. 优先级与实施顺序

### P0：先加可撤销 profiling（零语义改动）

记录以下时间并按 request/stage/chunk 输出 JSON：

1. Stage0/1/2 的 queue wait、runner execute、output processing、stage submit/route；
2. Code2Wav 的 `_encode_chunk`、CFM 每 step、HiFT、state split/stack/clone；
3. SHM serialize、lock/write、read/deserialize、通知等待；
4. scheduler `schedule/update_from_output`、orchestrator poll idle ratio、connector ready wait；
5. NPU graph hit/miss、host/device synchronize、KV block/preemption、HBM 峰值。

必须先用同一 A3 镜像、模型、commit、deploy、数据和 warmup 重跑三组基线；不能仅以日志中粗粒度 transfer `0.00 ms` 断言跨阶段无成本。

### P1：Code2Wav 工作区复用（首个代码补丁）

按 `(batch, chunk_frames, prompt/cache length, dtype, device, terminal)` 建立固定 shape workspace pool：

- 复用 CFM `timeline`、CFG 输入和 estimator CNN/attention output buffer；
- 用 view/index 替代可证明安全的临时 cat/split；保留 conditional/unconditional 行顺序；
- 复用 HiFT 历史窗口和输出 staging buffer；每个 request 仍拥有独立状态，terminal/epoch 时明确清空；
- NPU 不支持的 in-place 或 graph 方式保留原 fallback。

验收必须先通过 `tests/model_executor/models/minicpmo_4_5/test_code2wav_batching.py`，再做 Seed-TTS 和 duplex。任何逐样本波形漂移、时长变化、PCM 断裂或状态串扰都回退。

### P2：NPU 固定 shape graph/launch 优化

在 P0/P1 确认 CFM/HiFT 占比后，只对稳定 bucket 做 NPU 图捕获或批量 launch；动态 token 数、终端 chunk、full-duplex boundary 走 eager fallback。图优化不得改变 dtype、CFG 公式、step 顺序和随机噪声来源。

### P3：SHM/ChunkTransfer 零拷贝与合并唤醒

只在同机 `SharedMemoryConnector` 做 header + tensor bytes/ring 的实验；保留当前 dict/序列化 fallback。先确认 `.tolist()`、msgpack、lock 文件和 connector wait 各自占比，再决定是否合并多个 chunk；不得调整 chunk boundary 作为“隐含优化”。

### P4：Scheduler/Orchestrator ready queue

把全量 replica 1ms poll 改为事件/ready queue 驱动的可配置路径；无 pending connector、无 output signal 时不创建请求包装对象。用 scheduler 状态机测试证明 `WAITING_FOR_INPUT`、`WAITING_FOR_CHUNK`、RUNNING、KV ready、abort、terminal marker 和 duplex fences 完全一致。

### P5：Stage1/bridge 小对象与 CPU copy

缓存特殊 token/boundary id，预分配有限长度 codec history，合并 metadata 张量构造；先只优化 simplex，随后用相同代码路径验证 duplex。不得删除 transcript、epoch、turn、segment_end/turn_end 或音频 codes 的 copy 隔离。

### P6：参数实验（每项独立 A/B/A）

按 `10→3→10` 的 `token2wav_n_timesteps` 开始；然后分别测试 `token2wav_float16`、chunk/context、各 stage capacity、graph/eager。参数实验不能和 P1–P5 代码补丁混跑，否则无法归因。若 P1 已减少 allocator/launch，重新建立该代码版本的参数基线。

## 5. 测试矩阵与验收门槛

### 静态/CPU 门禁

```bash
python -m compileall vllm_omni
pytest -q \
  tests/model_executor/models/minicpmo_4_5/test_code2wav_batching.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_async_chunk.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py \
  tests/distributed/omni_connectors/test_shm_connector.py \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  tests/core/sched/test_omni_ar_scheduler_streaming.py \
  tests/engine/test_orchestrator_stage_input_bridge.py \
  tests/engine/test_orchestrator_error_handling.py
```

已有 bridge 定向测试曾出现“27 passed 后进程以 `corrupted size vs. prev_size` 退出”的环境/清理问题；后续不能只报告测试数量，必须保存完整 stdout/stderr 并确认退出码。

### A3 普通流式性能

在同一镜像和普通 deploy 下固定 `(1,32)/(4,64)/(8,128)`，至少 warmup 后重复三次，保存原始 JSON 和完整日志。报告吞吐、TTFT、audio TTFP、E2EL、audio RTF、P50/P99（脚本提供时）、失败率、音频时长、PCM 连续性、HBM/NPU 利用率。每个候选采用 `A-B-A`，只改变一个变量。

### 精度

使用完全相同的 Seed-TTS、Daily-Omni、Video-MME 数据、成功请求分母和 sampling：WER/SIM、Daily、Video、无 PCM/请求失败/解析失败都不能有可测回退；Daily 当前余量只有 0.03pp，任何回退都视为失败。

### Full-duplex

使用 `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`，覆盖持续 PCM16、listen/speak、barge-in、跨 turn、turn/epoch/fence、事件顺序、音频分块连续性、断线/恢复。至少复现当前 50 sessions × 4 turns 的辅助测量，并保存 WER/SIM、TTFP、RTF、无 PCM、失败、连续性；不能用 simplex 结果替代。

### 通过条件与回滚

候选必须满足：

1. 目标指标在同设备、同并发、同数据上重复改善，且能由 profiling 解释；
2. 文本/多模态/语音精度无可测回退，赛事硬门槛不下降；
3. full-duplex 无事件乱序、fence/epoch/turn 错配、barge-in 失效、PCM 断裂、请求泄漏；
4. 无 OOM、KV preemption 异常、NPU graph fallback 风暴和吞吐反向退化。

任一门槛失败，恢复该候选的独立 patch/overlay，保留失败结果、日志和原因，不改默认 deploy。所有改动使用最小 NPU/同机分支，通用 CUDA/ROCm 路径保持原状。

## 6. 当前结论

最值得先做的不是继续盲调参数，而是 **P0 profiling → P1 Code2Wav CFM/HiFT workspace 复用 → 独立 `n_timesteps` A/B/A → P3 SHM → P4 scheduler/orchestrator**。原因是现有 A3 数据已显示 Stage2 是主要 wall-time 消耗，而 CFM 的 per-step allocation/cat 与每 request 状态复制是明确的核心代码开销；SHM 和 scheduler 优化只有在分层计时证明其占比后才值得投入。`token2wav_n_timesteps=3` 仍是 AGENT 指定的第一参数假设，但必须和核心代码优化分开验证，绝不能以参数下降掩盖推理引擎链路问题。

**P6-A 的实验 worktree 没有修改 Python 推理代码，只通过独立配置覆盖验证参数。** 实验期间主分支默认配置保持不变；后续是否合入由精度/全双工门禁决定。

## 7. 连续实验记录

第一轮已执行 P1 request-state clone → detached view 候选：性能三组已完成，Seed-TTS 全量精度正在后处理。后续实验必须等待该轮结果落盘，按 `challenge_docs/change_log/20260816_0555_experiment_queue.md` 的顺序串行推进；每轮保留原始 tmux 日志和独立结果目录，不覆盖 `origin/` baseline。

P6-A 已完成 `token2wav_n_timesteps=3` 的三组性能压测（P1 代码已恢复为 baseline，未混合两个变量）。初步日志显示 3 步使 Stage2 音频生成耗时显著下降，8/128 的 audio RTF 已从 baseline 约 2.62 降至约 0.97；这验证了 Stage2 CFM loop 是强热点，但因为它改变 ODE 递推精度，必须等待全量 Seed-TTS WER/SIM 后再决定是否进入后续 float16/chunk 实验。

### P6-A 最终结果（2026-08-16）

全量 Seed-TTS 已完成（2020 条、并发 4、pytest `1 passed`）。`n_timesteps=3` 的性能收益明确，但 WER 从 1.3875% 回退到 1.4542%，SIM 从 0.848603 回退到 0.846267；因此该参数按精度门禁拒绝，配置已恢复 `10`。这次结果进一步确认 Stage2 CFM loop 是主要 wall-time 热点，但不能以减少 ODE 步数换取精度损失。后续转向不改变递推步数的 dtype、workspace、bridge 和调度路径实验。

2026-08-17，按用户明确要求，P6-A 以合并提交 `8555eeb9` 合入 `minicpm-challenge`，当前默认配置为 `3`。该决定保留上述已测 WER/SIM 回退和未执行 full-duplex 验证的限制，属于显式性能/质量折中，不代表通过本项目的无损精度门禁。

### P6-B 结果（2026-08-16）

`token2wav_float16=true` 在独立 worktree `exp/p6b-float16` 中验证。服务成功启动，
但 Stage2 CFM 首个请求即因 `Input type (float) and bias type (c10::Half) should be
the same` 崩溃；性能组 1 的 32/32 请求失败，Seed-TTS 仅 4/2020 完成、2016 失败、
4 条无 PCM，WER/SIM 无法评估。第一次 B 启动还记录了 PYTHONPATH 覆盖导致的 `acl`
导入失败，已单独归档。结论：当前 float16 选项不是可用的参数优化，恢复默认 false；
若以后要做 dtype 优化，必须先在 Code2Wav CFM/HiFT 全链路建立显式输入、权重和 cache
dtype 归一化，并以单元测试验证后再重新测性能。

### P6-C chunk20 结果（2026-08-16）

独立 worktree `exp/p6c-chunk20`（提交 `0a61a41d`）只把
`codec_chunk_frames` 从 25 改为 20，`token2wav_n_timesteps=10` 保持不变。Chapter
10 三档性能均退化：相对 AGENT.md A3 基线，吞吐为 `-0.27%/-5.98%/-16.05%`，
audio RTF 为 `-0.31%/+5.02%/+17.72%`（1/32、4/64、8/128）；E2EL 也全面上升。
全量 Seed-TTS（2020/2020，失败/无 PCM/ASR/SIM 错误均 0）得到 WER `1.3921%`
（基线 1.3875%，+0.0046pp）、WavLM SIM `0.848174`（基线 0.848603，下降
0.000429），总音频时长 10231.0s。虽然请求门禁通过，性能和质量均不满足无回退，
因此正式配置保留 25 帧；完整记录见
`challenge_docs/change_log/20260816_0840_chunk20.md`，原始日志和 JSON 不覆盖基线。

由于 20 帧是单点回退，已保留相邻点 `exp/p6c-chunk30`（提交 `33ddf4ce`）继续测量，
避免把单点结果误判为所有 chunk 边界均不可优化。后续 context0/context6 实验仍须
独立验证 PCM 连续性、音频时长和 full-duplex fence/turn 语义。

### P6-C chunk30 结果（2026-08-16）

独立 worktree `exp/p6c-chunk30`（提交 `33ddf4ce`）的 Chapter 10 三档性能明显改善：
吞吐 `0.5265/0.6330/0.7672 req/s`，audio RTF `0.4612/1.5572/2.5455`，相对
A3 基线分别改善约 `+6.30%/+10.18%/+2.25%` 和 `-6.54%/-9.37%/-3.01%`，且
128/128 请求均完成。但全量 Seed-TTS 在约 395/2020 请求时触发跨阶段硬错误：
Stage2 `SharedMemoryConnector` 报 `MessagePack data is malformed: trailing characters
(byte 1)`，stage-1 EngineCore 随即退出，后续请求无法连接服务。该结果证明 chunk
边界改变会暴露现有 SHM 消息边界/并发稳定性风险，性能改善不能抵消链路崩溃；chunk30
拒绝，正式配置继续为 25。详细证据见
`challenge_docs/change_log/20260816_090734_chunk30.md` 和对应 run log。

Seed-TTS 收尾 JSON 进一步记录 `completed=398/2020`、`failed=1622`、`no_pcm=2`，
只有 396 条有 WER/SIM（WER 1.3116%、SIM 0.849311），因此 pytest 表面的
`1 passed` 不构成质量通过。

这次失败直接调整了后续 P3 思路：先对 SharedMemoryConnector 的 MessagePack header、
长度校验、并发 writer/reader 和 chunk terminal marker 做只读 profiling 与故障复现，
再考虑零拷贝/ring 快速路径；任何实现必须先通过长时间 Seed-TTS 和 duplex 消息完整性
测试，不能以“更少 copy”或更高吞吐作为唯一验收。

### P6-D context0 结果（2026-08-16）

独立 worktree `exp/p6d-context0`（提交 `ec04185f`）将
`codec_left_context_frames` 从 3 改为 0。三档性能全面退化：吞吐
`0.4667/0.5683/0.6792 req/s`（相对 A3 `-5.78%/-1.09%/-9.49%`），audio RTF
`0.6294/2.0143/3.4166`（`+27.57%/+17.23%/+30.21%`）。Seed-TTS 在约 133/2020
请求再次触发同一 `MessagePack data is malformed: trailing characters (byte 1)`，
仅 137 成功、1883 失败、1 条无 PCM；context0 拒绝，默认 context=3 保持不变。
这说明边界上下文改动不仅不能改善性能，还会放大 SHM 消息完整性风险；context6
仍作为独立相邻点测试，但 P3 必须优先定位该共享内存协议错误。

### P6-D context6 结果（截至性能阶段）

独立 worktree `exp/p6d-context6`（提交 `c61778fb`）将
`codec_left_context_frames` 从 3 改为 6。1/32、4/64 的局部性能分别为吞吐
`0.5078/0.5780 req/s`、audio RTF `0.4262/1.5051`，但 8/128 仅开始约 2 个请求
就触发与 chunk30/context0 相同的 `SharedMemoryConnector` MessagePack trailing
characters 错误并杀死 stage-1 EngineCore。Seed-TTS 正在同一 worktree 串行执行，
无论精度结果如何，该候选已因 Chapter 10 稳定性门禁拒绝；完整日志保存在
`challenge_docs/run_log/p6d_context6_simplex_20260816_092700.log` 和
`challenge_docs/run_log/p6d_context6_seed_accuracy_20260816_092700.log`。

context0/context6/chunk30 的共同故障使 P3 的第一步收敛为只读协议 profiling：记录
SHM slot 的 header、payload length、MessagePack decode 边界、并发 writer/reader、
chunk terminal marker 及失败请求序号；在复现并证明边界后，才允许评估零拷贝或 ring
buffer。当前不修改默认 deploy，也不把局部 1/32、4/64 改善当作通过。

### P6-E Stage 2 capacity candidate（已建 worktree，待串行测试）

为验证 Stage 2 admission 是否成为并发 8 的排队瓶颈，建立独立
`exp/p6e-stage2-seqs8`（`699c1c99`），仅将 Stage 2 `max_num_seqs` 从 4 改为 8。
该变量不与 Stage 0/1 capacity、token cap 或 Code2Wav 参数混合；按默认 A1、B、默认
A2、Seed-TTS 串行测试。容量提高可能增加 Code2Wav activation/HBM、bucket 碎片和 SHM
并发压力，若出现 OOM、KV preemption、MessagePack 错误、PCM/精度回退，保留证据并
拒绝，不改正式配置。
context6 Seed-TTS 最终 JSON 记录 `completed=137/2020`、`failed=1883`、`no_pcm=1`，
服务在约 128 条后退出，pytest `1 failed`。这使 context6 与 context0/chunk30 的
SHM 稳定性证据闭环，不能因为局部吞吐或 SIM 样本均值较高而放宽门禁。

P6-E 的默认 A1 也在 `(8,128)` 只完成 `25/128`、失败 `103`，复现同一个 SHM
trailing-bytes 错误；`(1,32)`/`(4,64)` 虽完成且吞吐为 `0.517972/0.606788 req/s`，
但严格 A/B/A 已因 A1 失效而无效。Stage2 `max_num_seqs=8` 的 B 仍会执行，以区分容量
变量是否改变故障阈值，但不能用局部低并发结果接受该候选。

P6-E B 已完成：`(1,32)` 吞吐 `0.523692 req/s`、TTFT `320.074 ms`、E2EL
`1909.096 ms`、audio RTF `0.468456`；`(4,64)` 吞吐 `0.599433 req/s`、E2EL
`6570.309 ms`、audio RTF `1.636925`；两档均完整成功，但没有相对 A3 基线形成稳定
优势。`(8,128)` 仅完成 `27/128`、失败 `101`，约第 27 条触发同一 SHM
`MessagePack data is malformed: trailing characters (byte 1)`，之后 API 连接被拒绝。
把 Stage2 admission 从 4 放宽到 8 没有修复 key/生命周期竞争，反而在高并发保留了
服务退出；P6-E 已按稳定性门禁拒绝，A2/Seed 仅作为串行对照收尾。

默认 A2 随后完成 `(1,32)=0.513836 req/s`、`(4,64)=0.525208 req/s`，但 `(8,128)`
仅完成 `20/128`、失败 `108`，同样在第约 20 条触发 SHM trailing-bytes 并退出。
这使 P6-E 的 A/B/A 性能现场完整，但三次高并发均失败，容量放宽没有可接受收益；
Seed-TTS 仍仅用于归档收尾，不能改变拒绝结论。

P6-E Seed-TTS 也在约第 249 条复现 SHM 错误，最终请求 `136/2020` 成功、`1884` 失败、
无 PCM `2`；仅 `134` 条有 WER/SIM，WER `1.4155%`、SIM `0.848637`。虽然评估器对
已完成样本输出阈值通过，按全量请求门禁仍必须拒绝。Stage2 `max_num_seqs=8` 因而同时
未解决性能高并发失败与长测 SHM 崩溃，正式配置保持 `4`，后续重点转向 P3/P3B 的
共享内存协议与 generation key 假设。

### 长测失败的归因规则（复测现场补充）

当前失败不是单一原因，必须按故障层级区分，不能把 pytest 的 `1 failed` 直接等同于“代码改坏了”：

1. **基线与候选共同出现的 SHM 协议/并发故障**：默认 A1 在 `(8,128)` 完成 `25/128` 后也报 `MessagePack data is malformed: trailing characters (byte 1)`，随后 Stage-1 EngineCore 退出、请求返回 HTTP 500。P1 B 在相同阶段、相同错误下完成 `25/128`。因此这两次不能证明 Code2Wav detached-view 补丁引入回退；它们证明当前 `SharedMemoryConnector` key-only 发现、SHM 生命周期/写读边界在高并发下本身不稳定。P3 的 header+logical-length 实验针对该核心问题，但仍需完整 A/B/A、Seed-TTS 和 full-duplex 才能判断修复是否有效。
2. **确定性的实现/参数错误**：`token2wav_float16=true` 直接触发 CFM 输入 float 与 Half bias 的 dtype mismatch；这是补丁与现有 dtype contract 不一致，属于代码/配置不兼容，已拒绝。`n_timesteps=3` 则服务可运行但 WER/SIM 回退，属于质量门禁拒绝，不是运行时崩溃。
3. **可运行但不满足验收的性能/质量变化**：chunk20 性能和 Seed-TTS 均轻微退化；即使 pytest 请求全部成功，也必须按 AGENT.md 的性能、WER/SIM 和音频完整性门禁拒绝。
4. **测试环境或收尾问题**：首次单测缺少 `pytest-mock` 的 `mocker` fixture；P3 单测的 16 个断言均通过但解释器退出码 `134 (corrupted size vs. prev_size)`，这是清理阶段的进程/本机运行时问题，需单独记录，不能伪装成逻辑测试通过。另有一次 PYTHONPATH 覆盖导致 `acl` 导入失败，已通过统一保留 A3 环境路径的脚本规避。

Chapter 10 runner 的断言是故意的：任一请求未完成就会报告失败，避免用剩余成功请求的吞吐掩盖服务已经退出。因此当前结论应写成“运行失败/候选拒绝/环境异常”三类，而不是笼统归因；所有原始日志和 JSON 均保留在 `challenge_docs/run_log/`、`challenge_docs/batch_result/`。

P1 的严格 A/B/A 复测进一步验证了这一归因：默认 A1、P1 B、默认 A2 在 `(8,128)`
分别只完成约 `25/128`、`25/128`、`26/128`，三者都在同一个 SHM trailing-bytes 错误
后退出。因此当前没有证据把故障归因到 clone-view 改动本身；在 SHM framing 稳定前，P1
只能保留单次候选结果，不能合入默认路径。

P1 B 的 Seed-TTS A/B/A 补测最终生成 `completed=137/2020`、`failed=1883`、`no_pcm=1`，
约第 133 条触发同一 SHM trailing-bytes 错误；仅 136 条成功音频的 WER 均值为 `2.6308%`、
SIM `0.846211`，pytest 失败。单次性能/精度改善因此正式降级为“不可接受的候选记录”，
后续优先验证 SHM generation key，而不是继续优化 Code2Wav clone。

P3 的默认 A1 也复现基线：`(1,32)=0.494895 req/s`、`(4,64)=0.555624 req/s`，而
`(8,128)` 仅完成 `27/128`、失败 `101`，并在第约 27 条记录同一 trailing-bytes 后退出。
因此 P3 的 framing 改动尚未证明有效；候选 B 仍继续执行以确认是否改变故障阈值，若 B
或后续 Seed 不能全量成功，将转入 P3B 的 generation key 实验。

P3 B 的 `(1,32)/(4,64)` 完成且吞吐 `0.516469/0.587205 req/s`，但 `(8,128)` 仍为
`27/128`、`101` 失败，日志在第约 27 条明确记录 trailing-bytes。与默认 A1 的失败
阈值完全一致，说明只增加 payload header/读取 logical length 不足以解决旧 key 句柄与
unlink/recreate 竞态；P3 正式拒绝，下一步只改变 transfer generation key，不混合 chunk、
dtype 或调度参数。

### SHM 故障的下一层代码假设：key 重用窗口

沿代码链路继续追踪后发现，`try_send_via_connector()` 调用
`connector.put(..., put_key=req_id, ...)`，而同一个流式请求可能跨多个 chunk/Stage
重复发送。`SharedMemoryConnector.shm_write_bytes()` 在同名 segment 存在时会
`unlink()` 后重新创建；接收端可能已经通过同一 key 打开旧 segment，之后才取得锁。
锁文件能串行写入和读取，但不能让“旧 handle 与新 segment 的切换”原子化，因而存在读到
旧/新 payload 边界混合、随后 msgpack 报 trailing bytes 的窗口。当前 P3 只改变自描述
payload length，不改变 key 生命周期，必须把它作为独立假设验证；若 P3 仍失败，下一轮
应只测试 generation/transfer 唯一 key（或原子发布指针），不得同时改变 chunk、dtype 或
调度参数。该假设来自现有代码静态证据，最终以 P3 长测日志和最小复现确认。

### 复测环境卫生与队列编排（2026-08-16）

复测过程中发现两个会污染结论的测试基础设施问题：旧队列只等待前一个 tmux session，
曾让 P3B A1 和 P4 A1 同时占用 A3 NPU；另有异常退出留下约 34,561 个 benchmark
`/dev/shm` 文件，使 64 MiB tmpfs 达到 100%。两项均不是模型精度逻辑，但会造成服务
启动、SHM 分配和性能数字不可比。已停止受影响的测试、保留原始日志、清理确认属于
benchmark 的 `shm_*`/`chatcmpl-*`/`sem.*` 临时文件，并将 P4 及后续队列改为等待
P3B（含通过时的 full-duplex）后严格串行；每个性能 leg 收尾也执行同样的安全清理。

清理不能替代 generation-key 代码实验：它只保证 SHM 容量和 NPU 独占，P3B 仍需在
`p3b_clean_*` 目录完成 A/B/A、全量 Seed-TTS 和 full-duplex 门禁。
