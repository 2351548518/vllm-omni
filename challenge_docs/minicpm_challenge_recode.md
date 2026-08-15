
[2351548518/vllm-omni: A framework for efficient model inference with omni-modality models](https://github.com/2351548518/vllm-omni)

在宿主机（或已挂载 NPU 的容器）上确认驱动与设备正常：

```Shell
npu-smi info

free -h

ls -d /usr/local/Ascend/driver
cat /usr/local/Ascend/driver/version.info

tmux kill-session -t 0 && tmux kill-session -t 1 && tmux ls

pgrep -af 'vllm|MiniCPM|minicpmo'

npu-smi info
```

![image.png](https://raw.githubusercontent.com/2351548518/images/main/20260710/20260813212542758.png)

## CPU 和 NPU 监控

  终端 1：CPU、内存、进程和磁盘

```
btop
```

  这台环境已经安装了 btop。进入后可以按 / 搜索 pytest 或 VLLMStageEngi。

  终端 2：每秒刷新 NPU 状态

```
watch -n 1 'npu-smi info'
```


  重点观察：

  - AICore(%)：NPU 计算利用率
  - HBM-Usage(MB)：显存/HBM 使用量
  - Power(W)：功耗
  - Temp(C)：温度
  - Health：设备健康状态
  - 底部进程列表：占用 NPU 的 PID 与显存

  仅监控 CPU 排名前 20 的进程也可以使用：

```
watch -n 1 'ps -eo pid,ppid,%cpu,%mem,rss,etime,cmd --sort=-%cpu | head -n 20'
```

  建议布局：

  终端 1：运行 pytest
  终端 2：btop
  终端 3：watch -n 1 'npu-smi info'

## 安装 clash

```
apt-get update && apt-get install -y unzip
```

[nelvko/clash-for-linux-install: 😼 优雅地使用基于 clash/mihomo 的代理环境](https://github.com/nelvko/clash-for-linux-install)

## codex

[openai/codex: Lightweight coding agent that runs in your terminal](https://github.com/openai/codex)

## Git

```

git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

git config --global http.proxy socks5h://127.0.0.1:7890
git config --global https.proxy socks5h://127.0.0.1:7890

git config --global user.name "zeeore"
git config --global user.email "zeeoreone@outlook.com"



git config user.name; git config user.email


git push -u origin minicpm-challenge

git remote -v
```

```
git config --global --unset-all http.proxy
git config --global --unset-all https.proxy
```

```
 常用完整流程：

# 1. 查看改动
git status

# 2. 添加所有改动
git add -A

# 3. 创建提交
git commit -m "描述本次修改"

# 4. 拉取远端更新并把本地提交放到最新提交之后
git pull --rebase origin minicpm-challenge

# 5. 推送到远端
git push -u origin minicpm-challenge

以后该分支已经建立上游关联后，可以简化为：

git add -A
git commit -m "描述本次修改"
git pull --rebase
git push

仅拉取远端代码：

git pull origin minicpm-challenge

仅推送已有提交：

git push origin minicpm-challenge

如果 git commit 显示 nothing to commit，表示当前没有尚未提交的修改，可以直接执行 git
pull --rebase 和 git push。
```
## 安装 MiniCPM-o Token2Wav 依赖

MiniCPM-o 4.5 的语音输出（TTS）依赖 step-audio2 系列的 flow / HiFiGAN 声码器与音频 tokenizer 资源：

```Shell
#可更换pip源
pip install stepaudio2-minicpmo -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 
pip install step-audio2 -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-deps 
```

**说明**：为什么要用 `--no-deps`。step-audio2 的部分传递依赖会与镜像内已锁定的 torch / torch_npu 版本冲突，用 `--no-deps` 只装包体本身，复用镜像里的昇腾适配栈。


**说明**：在昇腾上，vLLM-Omni 会优先使用内置的 MiniCPMO45Token2wav（in-tree step_audio2_core 后端），而不是 stepaudio2 包里硬编码 `.cuda()` 的实现。上述依赖主要用于提供声码器权重加载所需的模块与资源。

---

RuntimeError: Failed to find C++ compiler

这是因为缺少 C++ 编译器。如果你后续需要用到 Triton 的 autotune 等功能，可能需要先安装：

```
apt-get install -y build-essential

apt-get install -y tmux
```
## 安装 vLLM-Omni


```Shell
# 官方仓库
git clone https://github.com/vllm-project/vllm-omni.git -b minicpm-challenge
# 自己fork的仓库
git clone https://github.com/2351548518/vllm-omni.git -b minicpm-challenge

cd vllm-omni

git checkout minicpm-challenge

SETUPTOOLS_SCM_PRETEND_VERSION=0.25.0 pip install -e .  -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## 模型权重

目前机器上已挂载模型，可直接使用 `/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5`

## 启动服务

MiniCPM-o 4.5 是 thinker + talker + Token2Wav 的多阶段流水线。仓库自带部署配置 `vllm_omni/deploy/minicpmo_4_5.yaml` 已包含 `platforms.npu` 覆盖项，`--omni` 会自动加载：

```Shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni \
    --served-model-name openbmb/MiniCPM-o-4_5 \
    --trust-remote-code \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --stage-init-timeout 600 \
    --host 0.0.0.0 --port 8091 \
    2>&1 | tee "$LOG_DIR/serve_main_${RUN_ID}.log"
```


关于 stage 与 NPU 设备分配，可编辑 `vllm_omni/deploy/minicpmo_4_5.yaml` 中的 `platforms.npu.stages`：

```YAML
platforms:
  npu:
    stages:
      - stage_id: 0
        max_num_batched_tokens: 8192
        compilation_config:
          cudagraph_mode: PIECEWISE
      - stage_id: 1
        max_num_batched_tokens: 8192
        compilation_config:
          cudagraph_mode: PIECEWISE
```

每个 stage 通过 `devices: "0"` 指定 NPU 序号。多卡时按物理卡号分配，例如 thinker 用 `"0"`，talker 用 `"1"`。


首次启动会编译图并加载声码器，`--stage-init-timeout 600` 用于给足初始化时间。

可用 `--deploy-config` 切换不同卡数布局：

- `minicpmo_4_5.yaml`：1 卡。thinker 与 talker+Token2Wav 共用 NPU 0
- `minicpmo_4_5_2gpu.yaml`：2 卡。thinker 在 NPU 0，talker+Token2Wav 在 NPU 1
- `minicpmo_4_5_3gpu.yaml`：3 卡。thinker 2 路 TP（NPU 0/1），talker 在 NPU 2
- `minicpmo_4_5_8x4090.yaml`：8 卡。thinker 4 路 TP（NPU 0-3），talker 在 NPU 4

需要等待开放 8091 端口
## 在线服务验证

服务就绪后（日志出现监听端口），用第 6 节启动的服务做验证，端口为 8091。

### 7.1 文本冒烟测试


```Shell
set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

curl http://127.0.0.1:8091/v1/models 2>&1 | tee "$LOG_DIR/smoke_models_${RUN_ID}.log"

curl http://127.0.0.1:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"openbmb/MiniCPM-o-4_5",
    "messages":[{"role":"user","content":"用一句话介绍你自己"}],
    "modalities":["text"],
    "max_tokens":128}' \
    2>&1 | tee "$LOG_DIR/smoke_text_${RUN_ID}.log"
```


```
root@e334958f80b8:/workspace# curl http://127.0.0.1:8091/v1/chat/completions \//127.0.0.1:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"openbmb/MiniCPM-o-4_5",
    "messages":[{"role":"user","content":"用一句话介绍你自己"}],
    "modalities":["text"],
    "max_tokens":128}'
{"id":"chatcmpl-bf1cfb02025d3abe","object":"chat.completion","created":1785831550,"model":"openbmb/MiniCPM-o-4_5","choices":[{"index":0,"message":{"role":"assistant","content":"你好，我是Qwen，一个由阿里云开发的大型语言模型，我能够理解多模态信息，提供广泛的知识和信息帮助。","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"routed_experts":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":12,"total_tokens":44,"completion_tokens":32,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"prompt_text":null,"kv_transfer_params":null,"metrics":null}root@e334958f80b8:/workspace# 
```
### 7.2 多模态输入与语音输出（curl）

手写请求若要语音输出，`use_tts_template` 必须放在请求根部,不要放在`extra_body`中。curl 不会展开嵌套的 `extra_body`：

```Shell
set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

curl http://127.0.0.1:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model":"openbmb/MiniCPM-o-4_5",
    "messages":[{"role":"user", "content":"先打个招呼，再用一句话介绍 vLLM。"}],
    "modalities":["text","audio"],
    "chat_template_kwargs":{"use_tts_template":true}
    }' \
    2>&1 | tee "$LOG_DIR/smoke_tts_${RUN_ID}.log"
```

返回音频为 base64 编码的 24 kHz 单声道 WAV，字段路径为 `choices[].message.audio.data`。

### 7.3 OpenAI Python 客户端

```
pip install httpx[socks]  -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 
```

```Shell
cd examples/online_serving/minicpmo
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

python openai_chat_completion_client_for_multimodal_generation.py --query-type use_image --host localhost --port 8091 2>&1 | tee "$LOG_DIR/online_client_image_${RUN_ID}.log"
python openai_chat_completion_client_for_multimodal_generation.py --query-type text --modalities text --port 8091 --prompt "用一句话介绍你自己。" 2>&1 | tee "$LOG_DIR/online_client_text_${RUN_ID}.log"
```

注: 如果没有网络可能image下载失败，优先使用文本进行测试即--query-type text

### 7.4 Gradio Demo


```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

bash examples/online_serving/minicpmo/run_gradio_demo.sh 2>&1 | tee "$LOG_DIR/gradio_script_${RUN_ID}.log"
```

或：

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8091/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862 \
    2>&1 | tee "$LOG_DIR/gradio_python_${RUN_ID}.log"
```


打开 `http://<host>:7862`。取消勾选 Generate speech output (TTS) 即为纯文本回复。

### 7.5 输出模态控制


- `["text"]`：仅文本，不追加 TTS bos
- `["text", "audio"]` 或==不设置==：文本 + 24 kHz 语音

语音输出需要 `chat_template_kwargs.use_tts_template=true`。curl 放在请求根部；OpenAI Python SDK 可放在 `extra_body`，SDK 会合并到根部。

更完整说明： https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/minicpmo/README.md

### **7.6 跑 Seed-TTS 数据集**

用 `vllm bench serve` 对 Seed-TTS 做 TTS 吞吐 / 时延评测（文本 + 音频输出）。模型路径、服务端口与第 5 / 6 节一致：`/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5`、`8091`。

#### **7.6.1 数据集**

数据来源： https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval （含 `en/`、`zh/` 的 `meta.lst` 与 `prompt-wavs/`）。

```
pip install -U huggingface_hub -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org

pip install httpx[socks] -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

```
tar -xvf /workspace/shared_assets/datasets/CowboyZ/seed-tts-eval/seedtts_testset.tar -C /workspace/user_data/datasets/seed-tts-eval
```
#### **7.6.2 启动服务**

按第 6 节启动即可（Seed-TTS 默认把参考音频打成 inline base64，一般不必加 `--allowed-local-media-path`）：

```Shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"

RUN_ID=$(date +%Y%m%d_%H%M%S)

vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --stage-init-timeout 1000 \
  --host 0.0.0.0 \
  --port 8091 \
  2>&1 | tee "$LOG_DIR/serve_${RUN_ID}.log"
```
#### **7.6.3 发送评测请求**

若要算 WER，先装打分依赖（Whisper-large-v3 / Paraformer-zh + jiwer，首次运行会下载 ASR 模型）：

```
pip install jiwer==4.0.0 zhon==2.1.1 funasr==1.3.29 zhconv==1.4.3  -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 
```

服务就绪后执行：

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"

RUN_ID=$(date +%Y%m%d_%H%M%S)
export SEED_TTS_SIM_EVAL=1

vllm bench serve \
--omni \
--port 8091 \
--trust-remote-code \
--max-concurrency 1 \
--num-warmups 3 \
--dataset-name seed-tts \
--dataset-path /workspace/user_data/datasets/seed-tts-eval/seedtts_testset \
--num-prompts 32 \
--disable-shuffle \
--no-oversample \
--seed-tts-wer-eval \
--seed-tts-wer-save-items \
--model openbmb/MiniCPM-o-4_5 \
--tokenizer /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
--seed-tts-file-ref-audio \
--endpoint /v1/chat/completions \
--backend openai-chat-omni \
--percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
--extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}' \
--save-result \
--result-dir /workspace/user_data/result/seed-tts-eval \
2>&1 | tee "$LOG_DIR/bench_${RUN_ID}.log"
```

```shell
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```


说明：  

- `--dataset-name seed-tts`：走 Seed-TTS 数据模块；可用 `--seed-tts-locale en|zh` 选语种（默认 `en`）
- `--seed-tts-wer-eval`：**必加**，否则不保留合成音频 PCM，WER 完全不会计算
- `--seed-tts-wer-save-items`：在上一项基础上，结果 JSON 里额外保存逐条 ASR / WER 明细（键名 `seed_tts_wer_eval_items`）；单独加它不生效
- `--extra_body` 必须带 `"modalities": ["text", "audio"]` 与 `"use_tts_template": true`，否则不会走 TTS
- `--percentile-metrics` 中的 `audio_ttfp` / `audio_rtf` 用于看首包音频时延与实时率
- 若改用 `file://` 参考音频（`--seed-tts-file-ref-audio`），serve 需加 `--allowed-local-media-path /workspace/seed-tts-eval`

#### **7.6.4 评测结果参考**

上述命令（`en` 语种、32 条、并发 1）的实测数值：

| 指标             | 数值     |
| -------------- | ------ |
| Mean TTFT (ms) | 333.26 |
| Mean TTFP (ms) | 986.47 |
| RTF            | 0.44   |

自己测的

910C


```
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  82.50     
Request throughput (req/s):              0.39      
Peak concurrent requests:                2.00      
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2577.33   
Median E2EL (ms):                        2596.07   
P99 E2EL (ms):                           3356.65   
================== Text Result ===================
Total input tokens:                      4801      
Total generated tokens:                  481       
Output token throughput (tok/s):         5.83      
Peak output token throughput (tok/s):    23.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          64.03     
---------------Time to First Token----------------
Mean TTFT (ms):                          547.77    
Median TTFT (ms):                        546.64    
P99 TTFT (ms):                           790.99    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
================== Audio Result ==================
Total audio duration generated(s):       139.00    
Total audio frames generated:            3336000   
Audio throughput(audio duration/s):      1.68      
Streaming continuity OK rate:            100.00%   
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    1464.35   
Median AUDIO_TTFP (ms):                  1475.69   
P99 AUDIO_TTFP (ms):                     1717.79   
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.61      
Median AUDIO_RTF:                        0.60      
P99 AUDIO_RTF:                           0.80      
==================================================

==== Seed-TTS eval (seed-tts-eval protocol) =====      
Evaluated (WER, lower is better):        32        
Mean WER:                                0.0107    
Median WER:                              0.0000    
Request failed:                          0         
No PCM captured:                         0         
ASR / WER failed:                        0         
SIM evaluated (higher ~ closer):         32        
Mean SIM:                                0.8463    
Median SIM:                              0.8497    
SIM skipped (no ref path):               0         
SIM embedding errors:                    0         
==================================================
```

910B3

```
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  102.83    
Request throughput (req/s):              0.31      
Peak concurrent requests:                2.00      
----------------End-to-end Latency----------------
Mean E2EL (ms):                          3212.20   
Median E2EL (ms):                        2944.26   
P99 E2EL (ms):                           4679.16   
================== Text Result ===================
Total input tokens:                      4795      
Total generated tokens:                  476       
Output token throughput (tok/s):         4.63      
Peak output token throughput (tok/s):    30.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          51.26     
---------------Time to First Token----------------
Mean TTFT (ms):                          418.68    
Median TTFT (ms):                        382.89    
P99 TTFT (ms):                           741.43    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
================== Audio Result ==================
Total audio duration generated(s):       134.92    
Total audio frames generated:            3238080   
Audio throughput(audio duration/s):      1.31      
Streaming continuity OK rate:            100.00%   
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    1625.46   
Median AUDIO_TTFP (ms):                  1602.81   
P99 AUDIO_TTFP (ms):                     1957.66   
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.79      
Median AUDIO_RTF:                        0.79      
P99 AUDIO_RTF:                           1.11      
==================================================
==== Seed-TTS eval (seed-tts-eval protocol) =====
Evaluated (WER, lower is better):        32        
Mean WER:                                0.0269    
Median WER:                              0.0000    
Request failed:                          0         
No PCM captured:                         0         
ASR / WER failed:                        0         
SIM evaluated (higher ~ closer):         32        
Mean SIM:                                0.8433    
Median SIM:                              0.8529    
SIM skipped (no ref path):               0         
SIM embedding errors:                    0         
==================================================
```


TTFP 是首个音频包延迟，比 TTFT 多出的部分是 Talker + Token2Wav 的启动开销。RTF 0.44 表示合成速度约为实时的 2.3 倍。数值随硬件、并发和参考音频长度变化，仅供对照
### **7.7 跑 Daily-Omni 数据集**

Daily-Omni 用 MiniCPM 官方交错 packing（1fps 帧与 1s 音频交替）测视听 MCQ 准确率。模型仍用 `/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5`，服务端口 `8091`；数据放在 `/workspace/user_data/datasets/Daily-Omni`。

#### **7.7.1 下载数据集**

数据来源： https://huggingface.co/datasets/liarliar/Daily-Omni （需 `qa.json` 与 `Videos.tar`）。


```Shell
pip install -U huggingface_hub  -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 

pip install httpx[socks] -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 

hf download liarliar/Daily-Omni \
    qa.json Videos.tar \
    --repo-type dataset \
    --local-dir /workspace/user_data/datasets/Daily-Omni

cd /workspace/user_data/datasets/Daily-Omni

tar -xf Videos.tar
```

同样用 `hf` 而不是已废弃的 `huggingface-cli`。

#### **7.7.2 启动服务**

在第 6 节命令基础上增加 Daily-Omni 必需参数：`--interleave-mm-strings`（保证 image/audio 按时间交错）与 `--allowed-local-media-path`（允许 bench 以 `file://` 发送抽帧 JPEG / 分段 WAV）。

```Shell
cd /workspace/user_data/vllm-omni

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DAILY_OMNI_VIDEOS=/workspace/user_data/datasets/Daily-Omni/Videos

export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni \
    --served-model-name openbmb/MiniCPM-o-4_5 \
    --trust-remote-code \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --stage-init-timeout 600 \
    --host 0.0.0.0 --port 8091 \
    --allowed-local-media-path "${DAILY_OMNI_VIDEOS}" \
    --interleave-mm-strings \
    2>&1 | tee "$LOG_DIR/daily_omni_serve_${RUN_ID}.log"
```

说明：

- 不需要 `--media-io-kwargs '{"video":{...}}'`：`minicpm-interleave` 模式由客户端自己按 1fps 抽帧并切分音频，发给服务端的只有 JPEG 与 WAV 分段，服务端不会解码原始视频，该参数不生效
- deploy YAML 的 `limit_mm_per_prompt` 需满足 `image >= 64`、`audio >= 64`（仓库内 `minicpmo_4_5.yaml` 已是 64/64），交错帧数上限即 64
- deploy YAML 中 thinker 采样应对齐 OmniEvalKit Daily-Omni：`temperature: 0.0`、`repetition_penalty: 1.0`、`max_tokens: 128`（vLLM chat 无 HF `num_beams=3`，用 greedy 近似）。`repetition_penalty` 只能写在 deploy 配置里，请求体带了也会被丢掉
- 客户端抽出的帧与音频分段会缓存到 `${DAILY_OMNI_VIDEOS}/.minicpm_daily_omni_interleave/<video_id>/`，因此 `Videos/` 目录必须可写并预留额外磁盘；同时 bench 与 serve 必须在同一台机器（或共享同一文件系统），`file://` 才能被服务端读到

#### **7.7.3 发送评测请求**

服务就绪后执行：

```
pip install decord2==3.4.0  -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 
```

加载很慢，需要等待很长时间

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

vllm bench serve \
  --omni \
  --port 8091 \
  --max-concurrency 10 \
  --dataset-name daily-omni \
  --num-prompts 2000 \
  --trust-remote-code \
  --no-oversample \
  --temperature 0 \
  --output-len 128 \
  --daily-omni-input-mode all \
  --daily-omni-pack-mode minicpm-interleave \
  --daily-omni-video-dir /workspace/user_data/datasets/Daily-Omni/Videos \
  --daily-omni-qa-json /workspace/user_data/datasets/Daily-Omni/qa.json \
  --model openbmb/MiniCPM-o-4_5 \
  --tokenizer /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el \
  --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}' \
  --save-result --result-dir /workspace/user_data/result/Daily-Omni \
  2>&1 | tee "$LOG_DIR/daily_omni_bench_${RUN_ID}.log"
  
```

说明：

- `--daily-omni-pack-mode minicpm-interleave`：按 OpenBMB 配方打包交错 image/audio（接近官方 ~80% 设置）；不要用默认 `qwen` packing 测 MiniCPM-o
- `--daily-omni-input-mode all`：视频 + 独立 WAV 一起发
- `--num-prompts 1197`：`qa.json` 的全量条数；配合 `--no-oversample`，填更大的值也只会跑 1197 条
- `--percentile-metrics` 不带 `audio_ttfp` / `audio_rtf`：Daily-Omni 只出文本（`modalities: ["text"]`），音频指标恒为空
- `--temperature 0` / `--output-len 128`：与 OmniEvalKit `do_sample=False`、`max_new_tokens=128` 对齐
- `--extra_body` 用 `"modalities": ["text"]`（Daily-Omni 只评文本答案，不开 TTS）
- 日志末尾会打印 Daily-Omni MCQ Overall Accuracy；也可用 `--daily-omni-save-eval-items` 把逐条对错写入结果 JSON

小流量调试可加 `--daily-omni-inline-local-video`（base64 内嵌，无需 allowlist），但全量 1197 条不建议。

#### **7.7.4 评测结果参考**

全量 1197 条、并发 10 的实测准确率：

```Plain
=========== Daily-Omni accuracy (MCQ) ============
Overall Accuracy: 937/1197 = 78.28%
Submitted (gold present):                1197
Successful HTTP (GitHub denom.):         1197
Correct:                                 937
Accuracy (ratio, same as above):         0.7828
Skipped (no gold):                       0
HTTP failed (excl. from GitHub acc.):    0
Parsed OK but no A–D found:              2
```

分维度结果：

|                       |                  |
| --------------------- | ---------------- |
| QA 类型                 | 准确率              |
| Comparative           | 86.26% (113/131) |
| Inference             | 83.77% (129/154) |
| Reasoning             | 81.71% (143/175) |
| AV Event Alignment    | 76.05% (181/238) |
| Context understanding | 75.65% (146/193) |
| Event Sequence        | 73.53% (225/306) |


按视频时长：30s 78.36% (507/647)、60s 78.18% (550 条中 430 条)，长短视频基本持平。

判定口径：`Successful HTTP` 为分母（与 Daily-Omni 官方仓库一致），HTTP 失败不计入。`Parsed OK but no A–D found: 2` 指模型回复里没解析出选项字母，按错误计。该结果与官方 ~80% 的报告值接近；若明显偏低，优先排查是否漏了 `--interleave-mm-strings` 或用了默认 `qwen` packing。

实测

910C

```
============ Serving Benchmark Result ============
Successful requests:                     1197      
Failed requests:                         0         
Maximum request concurrency:             10        
Benchmark duration (s):                  6481.24   
Request throughput (req/s):              0.18      
Peak concurrent requests:                13.00     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          54007.41  
Median E2EL (ms):                        53500.69  
P99 E2EL (ms):                           71126.02  
================== Text Result ===================
Total input tokens:                      4349821   
Total generated tokens:                  3531      
Output token throughput (tok/s):         0.54      
Peak output token throughput (tok/s):    8.00      
Peak concurrent requests:                13.00     
Total Token throughput (tok/s):          671.69    
---------------Time to First Token----------------
Mean TTFT (ms):                          44766.26  
Median TTFT (ms):                        43964.83  
P99 TTFT (ms):                           61904.49  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4768.53   
Median TPOT (ms):                        4764.01   
P99 TPOT (ms):                           9443.07   
---------------Inter-token Latency----------------
Mean ITL (ms):                           4733.83   
Median ITL (ms):                         4047.04   
P99 ITL (ms):                            10406.49  
==================================================
=========== Daily-Omni accuracy (MCQ) ============
Overall Accuracy: 933/1197 = 77.94%
Submitted (gold present):                1197      
Successful HTTP (GitHub denom.):         1197      
Correct:                                 933       
Accuracy (ratio, same as above):         0.7794    
Skipped (no gold):                       0         
HTTP failed (excl. from GitHub acc.):    0         
Parsed OK but no A–D found:              2         

--- Accuracy by QA Type ---
AV Event Alignment: 179/238 = 75.21%
Comparative: 111/131 = 84.73%
Context understanding: 147/193 = 76.17%
Event Sequence: 227/306 = 74.18%
Inference: 127/154 = 82.47%
Reasoning: 142/175 = 81.14%

--- Accuracy by Video Category ---
Autos & Vehicles: 17/29 = 58.62%
Comedy: 15/25 = 60.00%
Education: 136/167 = 81.44%
Entertainment: 163/217 = 75.12%
Film & Animation: 39/54 = 72.22%
Gaming: 36/41 = 87.80%
Howto & Style: 87/107 = 81.31%
Music: 33/42 = 78.57%
News & Politics: 64/82 = 78.05%
Nonprofits & Activism: 13/14 = 92.86%
People & Blogs: 117/150 = 78.00%
Pets & Animals: 30/35 = 85.71%
Science & Technology: 70/89 = 78.65%
Sports: 94/122 = 77.05%
Travel & Events: 19/23 = 82.61%

--- Accuracy by Video Duration ---
30s Duration: 503/647 = 77.74%
60s Duration: 430/550 = 78.18%
==================================================
```

910B

```
============ Serving Benchmark Result ============
Successful requests:                     1197      
Failed requests:                         0         
Maximum request concurrency:             10        
Benchmark duration (s):                  4166.57   
Request throughput (req/s):              0.29      
Peak concurrent requests:                13.00     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          34721.29  
Median E2EL (ms):                        34125.42  
P99 E2EL (ms):                           47800.81  
================== Text Result ===================
Total input tokens:                      4349821   
Total generated tokens:                  3536      
Output token throughput (tok/s):         0.85      
Peak output token throughput (tok/s):    10.00     
Peak concurrent requests:                13.00     
Total Token throughput (tok/s):          1044.83   
---------------Time to First Token----------------
Mean TTFT (ms):                          28805.69  
Median TTFT (ms):                        28144.95  
P99 TTFT (ms):                           42034.58  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          3049.48   
Median TPOT (ms):                        3101.90   
P99 TPOT (ms):                           6647.17   
---------------Inter-token Latency----------------
Mean ITL (ms):                           3023.88   
Median ITL (ms):                         2482.80   
P99 ITL (ms):                            7490.73   
==================================================
=========== Daily-Omni accuracy (MCQ) ============
Overall Accuracy: 933/1197 = 77.94%
Submitted (gold present):                1197      
Successful HTTP (GitHub denom.):         1197      
Correct:                                 933       
Accuracy (ratio, same as above):         0.7794    
Skipped (no gold):                       0         
HTTP failed (excl. from GitHub acc.):    0         
Parsed OK but no A–D found:              1         

--- Accuracy by QA Type ---
AV Event Alignment: 180/238 = 75.63%
Comparative: 110/131 = 83.97%
Context understanding: 148/193 = 76.68%
Event Sequence: 226/306 = 73.86%
Inference: 127/154 = 82.47%
Reasoning: 142/175 = 81.14%

--- Accuracy by Video Category ---
Autos & Vehicles: 18/29 = 62.07%
Comedy: 15/25 = 60.00%
Education: 136/167 = 81.44%
Entertainment: 161/217 = 74.19%
Film & Animation: 38/54 = 70.37%
Gaming: 36/41 = 87.80%
Howto & Style: 89/107 = 83.18%
Music: 33/42 = 78.57%
News & Politics: 64/82 = 78.05%
Nonprofits & Activism: 13/14 = 92.86%
People & Blogs: 117/150 = 78.00%
Pets & Animals: 30/35 = 85.71%
Science & Technology: 71/89 = 79.78%
Sports: 93/122 = 76.23%
Travel & Events: 19/23 = 82.61%

--- Accuracy by Video Duration ---
30s Duration: 504/647 = 77.90%
60s Duration: 429/550 = 78.00%
==================================================
```

### **7.8 跑 Video-MME 数据集**

Video-MME 用 MiniCPM 官方「仅抽帧」配方（OmniEvalKit `videomme`，w/o subs）测视频 MCQ 准确率：最多 96 帧以 `image_url` 发出，不带音轨。模型仍用 `/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5`，服务端口 `8091`；数据默认走 Hugging Face `lmms-lab/Video-MME`，也可放到 `/workspace/user_data/datasets/Video-MME/Video-MME`。

#### **7.8.1 数据集**

数据来源：https://huggingface.co/datasets/lmms-lab/Video-MME （需 parquet QA + `videos_chunked_*.zip`；可选 `subtitle.zip`）。

```Shell
pip install datasets   -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 

pip install fastparquet -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org 
```

复制到本地

```bash
#!/usr/bin/env bash

set -Eeuo pipefail

SOURCE_DIR="/workspace/shared_assets/datasets/lmms-lab/Video-MME"
DEST_DIR="/workspace/user_data/datasets/Video-MME"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "错误：源目录不存在：$SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST_DIR")"
cp -a "$SOURCE_DIR" "$DEST_DIR"

echo "复制完成：$SOURCE_DIR -> $DEST_DIR"

```

说明：

- 全量 900 视频 / 2700 题；`--videomme-duration short|medium|long` 可只跑某一时长桶（各 900 题）

#### **7.8.2 启动服务**

在第 6 节命令基础上增加 `--allowed-local-media-path`（bench 默认以 `file://` 发送抽帧 JPEG）。`minicpm-frames` 只发图像、不发交错音频，**不必**依赖 `--interleave-mm-strings`（与 Daily-Omni 不同）；若 serve 已为 Daily-Omni 打开该开关，保留无妨。


```Shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VIDEOMME_ROOT=/workspace/user_data/datasets/Video-MME/Video-MME

export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni \
    --served-model-name openbmb/MiniCPM-o-4_5 \
    --trust-remote-code \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --stage-init-timeout 600 \
    --host 0.0.0.0 --port 8091 \
    --allowed-local-media-path "${VIDEOMME_ROOT}" \
    2>&1 | tee "$LOG_DIR/videomme_serve_${RUN_ID}.log"
```


说明：
- deploy YAML 的 `limit_mm_per_prompt.image` **必须 ≥ 96**，否则超过 64 帧的样本会 HTTP 400（`At most N image(s) may be provided`）。仓库内 `minicpmo_4_5.yaml` 已设 `image: 96`（兼顾 Daily-Omni 64 帧与 Video-MME 96 帧）
- 客户端抽帧缓存写在视频目录下：`${VIDEOMME_ROOT}/.../.minicpm_videomme_frames/`，目录需可写并预留数十 GB；bench 与 serve 须共享同一文件系统（`file://`）
- 冷启动会对每个视频抽最多 96 帧；首次全量 2700 条前会有较长 warm-up，属正常现象
- 不需要客户端侧 `--media-io-kwargs`：`minicpm-frames` 已在本地完成抽帧，服务端收到的是 JPEG

若视频与 Daily-Omni 媒体不在同一父目录，把 `--allowed-local-media-path` 设为能覆盖两者的公共根（例如 `/workspace`）。

#### **7.8.3 发送评测请求**

服务就绪后执行（配方对齐 OmniEvalKit MiniCPM `videomme`）：


```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

vllm bench serve \
  --omni \
  --port 8091 \
  --max-concurrency 4 \
  --dataset-name videomme \
  --dataset-path /workspace/user_data/datasets/Video-MME/Video-MME \
  --num-prompts 2700 \
  --trust-remote-code \
  --no-oversample \
  --disable-shuffle \
  --temperature 0 \
  --output-len 128 \
  --videomme-pack-mode minicpm-frames \
  --videomme-max-frames 96 \
  --videomme-duration all \
  --model openbmb/MiniCPM-o-4_5 \
  --tokenizer /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el \
  --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}' \
  2>&1 | tee "$LOG_DIR/videomme_bench_${RUN_ID}.log"
```


也可不预下载，把 `--dataset-path` 换成 `lmms-lab/Video-MME`（需能访问 HuggingFace / 镜像，且服务端 `allowed-local-media-path` 覆盖 HF 缓存目录）。

说明：
- `--videomme-pack-mode minicpm-frames`：OmniEvalKit `videomme`（`load_av=false`）；不要用默认 `video_url` 或 Daily-Omni 的 `minicpm-interleave` 来对标官方 70.4
- `--videomme-max-frames 96` / `--output-len 128` / `--temperature 0`：与 OmniEvalKit `max_frames`、`max_new_tokens`、`do_sample=False` 对齐
- `--num-prompts 2700`：全量题数；配合 `--no-oversample` 填更大也只会跑到数据集大小
- `--extra_body` 用 `"modalities": ["text"]`，且**不要**加 `"use_tts_template": true`（同 Daily-Omni，纯文本 MCQ）
- `--percentile-metrics` 不带 `audio_ttfp` / `audio_rtf`：本配方只出文本
- 日志末尾会打印 Video-MME Overall / by-duration / by-domain；可用 `--videomme-save-eval-items` 把逐条对错写入结果 JSON
- 改 `max_frames` / packing 后需清缓存：`find /workspace/Video-MME -type d -name '.minicpm_videomme_frames' -exec rm -rf {} +`

小流量调试：

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

vllm bench serve ... \
  --videomme-duration short \
  --num-prompts 8 \
  2>&1 | tee "$LOG_DIR/videomme_debug_${RUN_ID}.log"
```

或加 `--videomme-inline-local-video`（base64 内嵌，无需 allowlist）；全量 2700 条不建议。

Video-MME-Short + 音轨（OmniEvalKit `videomme_short`）可改为：  

```Shell
  --videomme-pack-mode minicpm-interleave \
  --videomme-max-frames 64 \
  --videomme-duration short
```

此时 serve 还需 `--interleave-mm-strings`，且 `limit_mm_per_prompt.audio >= 64`。  

#### **7.8.4 评测结果参考**

全量 2700 条、`minicpm-frames` / 96 帧 / w/o subs、并发 4 的实测：

| 指标               | 数值                    |
| ---------------- | --------------------- |
| Overall Accuracy | **69.96%**（1889/2700） |
| short            | 80.33%（723/900）       |
| medium           | 70.33%（633/900）       |
| long             | 59.22%（533/900）       |
| Successful HTTP  | 2700 / 2700           |

官方 MiniCPM-o 4.5 报告 **70.4**（w/o subs）；上表与之差约 0.4pp。判定口径：`Successful HTTP` 为分母。若大量 HTTP 400 且报 `At most N image(s)`，优先把 deploy YAML 的 `image` 提到 ≥ 96 并重启 serve。

自己测的

910C

```
============ Serving Benchmark Result ============                                                                                                              
Successful requests:                     2700                                                                                                                   
Failed requests:                         0                                                                                                                      
Maximum request concurrency:             4                                                                                                                      
Benchmark duration (s):                  10633.06                               
Request throughput (req/s):              0.25                                   
Peak concurrent requests:                9.00                                   
----------------End-to-end Latency----------------                              
Mean E2EL (ms):                          15750.62                               
Median E2EL (ms):                        16356.32                               
P99 E2EL (ms):                           23938.71                               
================== Text Result ===================                              
Total input tokens:                      16432670                               
Total generated tokens:                  5487                                   
Output token throughput (tok/s):         0.52                                   
Peak output token throughput (tok/s):    9.00                                   
Peak concurrent requests:                9.00                                   
Total Token throughput (tok/s):          1545.95                                
---------------Time to First Token----------------                              
Mean TTFT (ms):                          12108.82                               
Median TTFT (ms):                        13106.96                               
P99 TTFT (ms):                           21858.76                               
-----Time per Output Token (excl. 1st token)------                              
Mean TPOT (ms):                          3546.60                                
Median TPOT (ms):                        3031.03                                
P99 TPOT (ms):                           7194.90                                
---------------Inter-token Latency----------------                              
Mean ITL (ms):                           3528.08                                
Median ITL (ms):                         2675.83                                
P99 ITL (ms):                            7193.96                                
==================================================                              
============ Video-MME accuracy (MCQ) ============                              
Overall Accuracy: 1876/2700 = 69.48%                                            
Submitted (gold present):                2700                                   
Successful HTTP (denominator):           2700                                   
Correct:                                 1876                                   
Skipped (no gold):                       0                                      
HTTP failed:                             0                                      
Parsed OK but no A-D found:              0                                      

--- Accuracy by Duration ---                                                    
short: 723/900 = 80.33%                 
medium: 628/900 = 69.78%                
long: 525/900 = 58.33%   

--- Accuracy by Duration ---                                                                                                                            [11/216]
short: 723/900 = 80.33%                 
medium: 628/900 = 69.78%                
long: 525/900 = 58.33%                  

--- Accuracy by Domain ---              
Artistic Performance: 241/360 = 66.94%                                          
Film & Television: 254/360 = 70.56%                                             
Knowledge: 570/810 = 70.37%             
Life Record: 450/630 = 71.43%                                                   
Multilingual: 61/90 = 67.78%                                                    
Sports Competition: 300/450 = 66.67%                                            

--- Accuracy by Sub Category ---                                                
Acrobatics: 54/90 = 60.00%              
Animation: 61/90 = 67.78%               
Astronomy: 67/90 = 74.44%               
Athletics: 56/90 = 62.22%               
Basketball: 52/90 = 57.78%              
Biology & Medicine: 73/90 = 81.11%                                              
Daily Life: 60/90 = 66.67%              
Documentary: 62/90 = 68.89%             
Esports: 53/90 = 58.89%                 
Exercise: 59/90 = 65.56%                
Fashion: 64/90 = 71.11%                 
Finance & Commerce: 67/90 = 74.44%                                              
Food: 59/90 = 65.56%                    
Football: 71/90 = 78.89%                
Geography: 59/90 = 65.56%               
Handicraft: 74/90 = 82.22%              
Humanity & History: 45/90 = 50.00%                                              
Law: 65/90 = 72.22%                     
Life Tip: 68/90 = 75.56%                
Literature & Art: 62/90 = 68.89%                                                
Magic Show: 63/90 = 70.00%              
Movie & TV Show: 63/90 = 70.00%                                                 
Multilingual: 61/90 = 67.78%                                                    
News Report: 68/90 = 75.56%             
Other Sports: 68/90 = 75.56%                                                    
Pet & Animal: 70/90 = 77.78%                                                    
Stage Play: 74/90 = 82.22%              
Technology: 64/90 = 71.11%              
Travel: 64/90 = 71.11%                  
Variety Show: 50/90 = 55.56%            

--- Accuracy by Task Type ---                                                   
Action Reasoning: 171/285 = 60.00%                                              
Action Recognition: 220/313 = 70.29%                                            
Attribute Perception: 183/222 = 82.43%                                          
Counting Problem: 128/268 = 47.76%                                              
Information Synopsis: 270/323 = 83.59%                                          
OCR Problems: 102/139 = 73.38%                                                  
Object Reasoning: 296/454 = 65.20%                                              
Object Recognition: 271/354 = 76.55%                                            
Spatial Perception: 38/54 = 70.37%                                              
Spatial Reasoning: 44/56 = 78.57%                                               
Temporal Perception: 47/55 = 85.45%                                             
Temporal Reasoning: 106/177 = 59.89%                                            
==================================================       
```
## 8. 离线推理

仓库提供端到端离线脚本 `examples/offline_inference/minicpmo/end2end.py`，直接在进程内跑完 thinker → talker + Token2Wav，无需先启动服务。文本结果与 24 kHz WAV 会写入输出目录。

**说明**：离线脚本默认加载 `vllm_omni/deploy/minicpmo_4_5.yaml`，其中的 `platforms.npu` 覆盖项会在昇腾上自动生效。默认单卡布局把 thinker 与 talker+Token2Wav 共置于 NPU 0。如需拆卡，用 `--deploy-config` 指定 2/3 卡布局，见第 6 节。

```Shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd examples/offline_inference/minicpmo
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

bash run_single_prompt.sh 2>&1 | tee "$LOG_DIR/offline_single_prompt_${RUN_ID}.log"
```

等价命令：  

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

python end2end.py --query-type text --output-dir output_audio 2>&1 | tee "$LOG_DIR/offline_text_${RUN_ID}.log"
```

多模态输入：

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

python end2end.py --query-type use_image --image-path /path/to/image.jpg 2>&1 | tee "$LOG_DIR/offline_image_${RUN_ID}.log"
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav 2>&1 | tee "$LOG_DIR/offline_audio_${RUN_ID}.log"
python end2end.py --query-type use_video --video-path /path/to/video.mp4 2>&1 | tee "$LOG_DIR/offline_video_${RUN_ID}.log"
python end2end.py --query-type use_audio --modalities text 2>&1 | tee "$LOG_DIR/offline_audio_text_${RUN_ID}.log"
```

支持的 `--query-type`：text、use_image、use_audio、use_video、use_multi_audios、use_mixed_modalities。

多卡布局示例脚本（3 卡：thinker 2 路 TP + talker 独占一卡）：

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

bash run_single_prompt_tp.sh 2>&1 | tee "$LOG_DIR/offline_tp_script_${RUN_ID}.log"
```

等价命令（将 REPO 替换为 vllm-omni 仓库根目录）：

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

python end2end.py --query-type use_audio \
    --deploy-config REPO/vllm_omni/deploy/minicpmo_4_5_3gpu.yaml \
    --stage-init-timeout 300 \
    2>&1 | tee "$LOG_DIR/offline_tp_audio_${RUN_ID}.log"
```

如果超时可以尝试调整stage-init-timeout和init-timeout两个参数

**提示**：昇腾多进程务必先执行 `export VLLM_WORKER_MULTIPROC_METHOD=spawn`。

**提示**：提示词占位符使用 MiniCPM 风格：`(<image>./</image>)`、`(<audio>./</audio>)`、`(<video>./</video>)`。语音输出依赖助手前缀上的 `<|tts_bos|>`，脚本已自动处理。

**提示**：输出 WAV 固定为 24 kHz 单声道。

更完整离线示例： https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/minicpmo/README.md

## **9. 全双工（Full-Duplex）实时语音部署**

第 6~8 节是"一问一答"的半双工模式：客户端发完整请求，服务端返回完整回复。全双工模式下客户端持续上传麦克风 PCM，模型自己在约 1 秒一个的 model unit 边界上决定"继续听（listen）"还是"开口说（speak）"，因此支持说话过程中被打断（barge-in）。

**警告**：全双工是实验特性（experimental），走 `vllm_omni/experimental/fullduplex` 这条路径。不支持视频输入与音视频同步，也未做生产级的多会话容量、公平调度与故障恢复。昇腾上属于实验组合，建议先在单会话场景下验证。

### **9.1 与半双工的差异**

- 协议：WebSocket，不是 HTTP。端点为 `/v1/realtime?duplex=1`（OpenAI Realtime 协议投影）和 `/v1/duplex`（原生会话控制协议）
- 打断判定由模型做，浏览器/客户端不跑 VAD，也不需要发 `input_audio_buffer.commit`；客户端只要每 200 ms 持续发 `input_audio_buffer.append` 即可
- 输入音频固定为 16 kHz 单声道 16-bit PCM，输出仍为 24 kHz 单声道
- 需要一段参考音色 WAV（决定输出音色），仓库模型目录里自带 `assets/HT_ref_audio.wav`
### **9.2 前置准备**

在完成第 1~5 节（容器、Token2Wav 依赖、vLLM-Omni、模型权重）之后，补装 WebSocket 客户端库：

```Shell
pip install websockets -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

准备一个输入音频转成全双工要求的 16 kHz 单声道 PCM16：

```Shell
sudo apt update && sudo apt install -y ffmpeg
ffmpeg -i input.wav -ac 1 -ar 16000 -sample_fmt s16 -c:a pcm_s16le input_16k.wav
```

**说明**：客户端会严格校验，采样率、声道数、位宽任意一项不符都会直接报错退出。

### **9.3 启动全双工服务**

全双工用专门的部署配置 `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`：

```Shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni \
    --served-model-name openbmb/MiniCPM-o-4_5 \
    --trust-remote-code \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
    --stage-init-timeout 600 \
    --host 0.0.0.0 --port 8091 \
    2>&1 | tee "$LOG_DIR/duplex_serve_${RUN_ID}.log"
```


**说明**：全双工路由只在部署配置里显式写了 `session_mode: duplex` 时才注册。用第 6 节的 `minicpmo_4_5.yaml` 启动的服务不会有 `/v1/duplex` 端点，连 WebSocket 会直接失败。

该配置的关键项：

```YAML
base_config: minicpmo_4_5.yaml
pipeline: minicpmo_4_5
session_mode: duplex
active_stream_window: 1
duplex_session:
  idle_ttl_s: 300              # 会话空闲多久后回收
  disconnect_grace_s: 30       # 断线后允许重连恢复的宽限期
  resume_replay_ttl_s: 60      # 重连后可重放的事件保留时长
  max_pending_turns_per_session: 4
  max_sessions: 2              # 与下方 max_num_seqs 对齐
stages:
  - stage_id: 0
    max_num_seqs: 2
  - stage_id: 1
    max_num_seqs: 2
  - stage_id: 2
    max_num_seqs: 2
```


**提示**：`max_sessions` 必须与三个 stage 的 `max_num_seqs` 对齐。默认 2 表示最多两路并发全双工会话，调大时三处要一起改。

### **9.4** **确认全双工端点已注册**

服务启动后在日志里查路由表：

```Shell
grep -E "Route: /v1/(duplex|realtime)" /workspace/user_data/vllm-omni/challenge_docs/run_log/duplex_serve_*.log
```

正常应看到：

```Plain
Route: /v1/realtime, Endpoint: realtime_websocket
Route: /v1/duplex, Endpoint: duplex_websocket
```

只有 `/v1/realtime` 而没有 `/v1/duplex`，说明 `session_mode: duplex` 没生效，检查 `--deploy-config` 路径是否指向了 duplex 配置。

### **9.5** **命令行 Demo 验证**

模型仓库MiniCPM-o-4_5/assets下自带一些ref_audio可以直接引用：

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url 'ws://localhost:8091/v1/realtime?duplex=1' \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav input_16k.wav \
    --ref-audio /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
    --output-dir /tmp/minicpmo_realtime_duplex_demo \
    2>&1 | tee "$LOG_DIR/duplex_cli_${RUN_ID}.log"
```


常用参数：

- `--chunk-ms`：上传分片时长，默认 200 ms，与浏览器行为一致
- `--timeout-s`：等待模型输出的超时，默认 60 秒。昇腾首次跑建议放大
- `--no-realtime-pacing`：关闭实时节奏，尽可能快地灌完音频，用于压测而非延迟测量
- `--require-audio`：要求本次必须产出音频，否则判定失败。模型选择 listen 属于正常行为，常规验证不要加这个参数

输出目录内容：

- `output.wav`：所有音频分片拼接后的 24 kHz 完整回复
- `audio_chunks/chunk_XXXX.wav`：逐个音频分片，用于看流式颗粒度
- `events.jsonl`：完整的 Realtime 事件流，排查协议问题时看这个
- `result.json`：结论与延迟指标

`result.json` 关键字段：


```JSON
{
  "ok": true,
  "model_decision": "speak",
  "audio_chunk_count": 12,
  "output_sample_rate_hz": 24000,
  "latency": {
    "ttft_ms": 320.5,
    "ttfp_ms": 480.2,
    "rtf": 0.42,
    "measurement_origin": "input_audio_buffer.commit send"
  },
  "transcript": "...",
  "errors": []
}
```

- `model_decision`：本轮模型决策，`speak` 为开口回复，`listen` 为选择继续听
- `ttft_ms` / `ttfp_ms`：首字延迟 / 首个音频包延迟
- `rtf`：生成耗时与音频时长之比，小于 1 才能实时播放

**说明**：`model_decision` 为 `listen`、`audio_chunk_count` 为 0 并不代表部署失败。只要 `ok` 为 `true` 且 `errors` 为空，就说明链路是通的，只是模型认为这段输入不需要回复。换一段语义完整的语音再试。

### **9.6** **浏览器实时对话**

带麦克风采集与播放的网页客户端，它自身提供页面并把同源 WebSocket 代理到后端：

```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

python -m examples.online_serving.minicpmo.realtime_web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8091 \
    --model openbmb/MiniCPM-o-4_5 \
    --ref-audio /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
    2>&1 | tee "$LOG_DIR/realtime_web_${RUN_ID}.log"
```


打开 `http://<host>:7862/`，允许浏览器使用麦克风后即可开始连续对话，说话可以打断模型正在播放的回复。

若前面挂了反向代理且代理不转发 WebSocket 升级请求，把浏览器直接指向单独暴露的 Realtime 地址：


```Shell
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)

python -m examples.online_serving.minicpmo.realtime_web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8091 \
    --ref-audio /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
    --public-realtime-url wss://public.example/v1/realtime \
    2>&1 | tee "$LOG_DIR/realtime_web_public_${RUN_ID}.log"
```

**警告**：容器要额外映射 7862 端口（`docker run` 加 `-p 7862:7862`）。另外浏览器只在 `https` 或 `localhost` 下才授予麦克风权限，远程访问需要配 HTTPS 或用 SSH 端口转发到本地。

### **9.7** **自定义客户端要点**

自己实现客户端时，按下面的事件契约来：

上行（客户端 → 服务端）：

- `session.update`：设置 `input_audio_format: pcm16` 与参考音色
- `input_audio_buffer.append`：每 200 ms 一片 base64 PCM16，会话开着就一直发，模型播放期间也不要停
- `playback.ack`：回报已播放到的位置，服务端据此推进历史提交
- `session.close`：结束会话

下行（服务端 → 客户端）：

- `response.created`：一次可见回复的第一个事件
- `response.speak` / `response.listen`：模型的说/听决策。`response.listen` 可能不伴随 `response.created`，不要据此认为有回复丢失
- `response.audio.delta`：一片有序音频，每片后面必定跟一个 `response.audio_transcript.delta`
- `response.audio.done` / `response.audio_transcript.done` / `response.done`：终止事件，`response.done` 每个回复恰好一次

**提示**：不要自己在客户端做 VAD 打断。打断由模型在 model unit 边界决定，客户端强行断流反而会破坏会话状态。

**提示**：完整协议契约见 `vllm_omni/experimental/fullduplex/DESIGN.md`。

## 10.Baseline测试


```
# 如有 需要 可以改成 DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
```


```
#先安装好pytest
pip install pytest-asyncio -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
#dailyomni数据处理依赖的包
pip install "datasets==3.6.0" -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
#seedtts中wer评估依赖的包
pip install jiwer zhon -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### 精度测试


---

单项测试(VideoMME)

---

```
cd /workspace/user_data/vllm-omni

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail

LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"

RUN_ID=$(date +%Y%m%d_%H%M%S)

env \
  ASCEND_RT_VISIBLE_DEVICES=0 \
  VLLM_TEST_MINICPMO_4_5_MODEL=/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  VLLM_DAILY_OMNI_QA_JSON=/workspace/user_data/datasets/Daily-Omni/qa.json \
  VLLM_DAILY_OMNI_VIDEO_DIR=/workspace/user_data/datasets/Daily-Omni/Videos \
  VLLM_SEED_TTS_DATASET_PATH=/workspace/user_data/datasets/seed-tts-eval/seedtts_testset \
  VLLM_VIDEOMME_DATASET_PATH=/workspace/user_data/datasets/Video-MME/Video-MME \
  ACC_BENCH_DAILY_OMNI_MAX_CONCURRENCY=2 \
  ACC_BENCH_SEED_TTS_MAX_CONCURRENCY=2 \
  ACC_BENCH_VIDEOMME_NUM_PROMPTS=2700 \
  ACC_BENCH_VIDEOMME_MAX_CONCURRENCY=4 \
  ACC_BENCH_VIDEOMME_DURATION=all \
  ACC_BENCH_MIN_VIDEOMME_ACCURACY=0.68 \
  SEED_TTS_WER_EVAL=1 \
  SEED_TTS_SIM_EVAL=1 \
  ACC_BENCH_RESULT_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/origin/accuracy \
  pytest -s -v -rs \
    tests/e2e/accuracy/minicpmo_4_5/test_minicpmo_4_5.py \
    -m 'full_model' \
    -k 'daily_omni_accuracy_bench' \
    --run-level full_model \
    2>&1 | tee "$LOG_DIR/baseline_accuracy_${RUN_ID}.log"
```

---

全部测试

```
cd /workspace/user_data/vllm-omni

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail

LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"

RUN_ID=$(date +%Y%m%d_%H%M%S)

env \
  ASCEND_RT_VISIBLE_DEVICES=0 \
  VLLM_TEST_MINICPMO_4_5_MODEL=/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
  VLLM_DAILY_OMNI_QA_JSON=/workspace/user_data/datasets/Daily-Omni/qa.json \
  VLLM_DAILY_OMNI_VIDEO_DIR=/workspace/user_data/datasets/Daily-Omni/Videos \
  VLLM_SEED_TTS_DATASET_PATH=/workspace/user_data/datasets/seed-tts-eval/seedtts_testset \
  VLLM_VIDEOMME_DATASET_PATH=/workspace/user_data/datasets/Video-MME/Video-MME \
  ACC_BENCH_DAILY_OMNI_MAX_CONCURRENCY=2 \
  ACC_BENCH_SEED_TTS_MAX_CONCURRENCY=2 \
  ACC_BENCH_VIDEOMME_NUM_PROMPTS=2700 \
  ACC_BENCH_VIDEOMME_MAX_CONCURRENCY=4 \
  ACC_BENCH_VIDEOMME_DURATION=all \
  ACC_BENCH_MIN_VIDEOMME_ACCURACY=0.68 \
  SEED_TTS_WER_EVAL=1 \
  SEED_TTS_SIM_EVAL=1 \
  ACC_BENCH_RESULT_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/origin/accuracy \
  pytest -s -v -rs \
    tests/e2e/accuracy/minicpmo_4_5/test_minicpmo_4_5.py \
    -m 'full_model' \
    --run-level full_model \
    2>&1 | tee "$LOG_DIR/baseline_accuracy_${RUN_ID}.log"
```


### 性能测试

```
# simplex_performance
# 跑（-k 过滤 A3 格；结果写 $BENCHMARK_DIR）
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
BENCHMARK_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/origin/simplex_performance python -m pytest -s -v \
  tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file /workspace/user_data/vllm-omni/challenge_docs/test_yaml/test_minicpmo_4_5.json \
  -k test_minicpmo_4_5_a2_challenge \
  2>&1 | tee "$LOG_DIR/baseline_simplex_performance_${RUN_ID}.log"
```

### 全双工测试

```Plain
# duplex_performance
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

set -o pipefail
LOG_DIR=/workspace/user_data/vllm-omni/challenge_docs/run_log
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
BENCHMARK_DIR=/workspace/user_data/vllm-omni/challenge_docs/batch_result/origin/duplex_performance \
pytest -s -v \
tests/dfx/perf/scripts/run_benchmark.py \
--test-config-file tests/dfx/perf/tests/test_minicpmo_4_5_duplex_seed_tts.json \
2>&1 | tee "$LOG_DIR/baseline_duplex_performance_${RUN_ID}.log"
```
