# vLLM-Omni 子赛道评测踩坑经验分享

> 环境:昇腾 910C 单卡 + 官方镜像 `vllm-omni:v0.25.0-a3`
> 面向:vLLM-Omni 子赛道参赛队伍

---

## 🤖 AI 总结(by AI)

> 在官方 vLLM-Omni 评测环境中有 3 个最容易被忽略的坑:①官方 pytest 文档参数 `--num-warmup` 系笔误(实为 `--num-warmups`);②Seed-TTS 的 SIM 指标需要 `--seed-tts-file-ref-audio` 指定参考音频,否则会被静默跳过;③WER 评测联网时初始化会访问 HuggingFace 导致卡死,离线环境必须加 `HF_HUB_OFFLINE=1`。此外 `connector_get_sleep_s` 与 `codec_chunk_ramp` 两个配置在 minicpmo 模型上无消费端,配置了也不生效,不必浪费时间。

---

## 📋 完整内容

各位老师好,我们在 910C + 官方镜像上跑 vLLM-Omni 子赛道评测时踩了一些文档和环境层面的坑,整理出来分享,说的不对的地方还请海涵:

### 1. 官方 pytest 文档笔误
`--num-warmup`(单数)实际 CLI 参数是 `--num-warmups`(复数),按文档原样抄会报参数错误。

### 2. Seed-TTS 的 SIM 会被静默跳过
设了 `SEED_TTS_SIM_EVAL=1` 之后,还需要加 `--seed-tts-file-ref-audio` 指定参考音频路径,否则 ref 路径为空,SIM 整列静默缺失——不是跑挂了,是没喂参考音频。

### 3. WER 评测联网会卡死
WER 初始化会访问 HuggingFace 拉模型,网络不通时整个评测卡住不动;离线环境记得加 `HF_HUB_OFFLINE=1`。

### 4. 两个"改了也没用"的配置
- `connector_get_sleep_s`:minicpmo 链路里没有 Python 消费端,实际轮询间隔硬编码 1ms,改配置零效果
- `codec_chunk_ramp`:目前只适配 Qwen3-TTS,minicpmo 模型无消费端

### 5. 评测客户端偶发崩溃
全部请求跑完后,客户端偶尔报 `corrupted size vs. prev_size`(glibc teardown 阶段崩溃),不影响已产出的结果,重跑一次即可。

### 6. serve 启动路径
建议从 `/tmp` 目录启动 `vllm serve`,避免在 `/vllm-workspace` 目录下启动时遮蔽 vllm 包 import。

### 7. 镜像内装依赖
pip 换阿里云源后,`pip install stepaudio2-minicpmo`,再 `pip install step-audio2 --no-deps`(必须 `--no-deps`,防止覆盖镜像内 torch/torch_npu 把 NPU 环境装坏)。

---

祝大家评测顺利,一起加油!🚀

*(本文档由 AI 辅助整理)*
