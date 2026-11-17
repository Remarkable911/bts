# BTS

BTS 是一种面向长上下文大模型推理的分层稀疏注意力方法，支持最长 128K 上下文。
它先在块级别筛选重要区域，再在候选块内进行细粒度的词元选择，并保留开头、最近邻和
末尾查询所需的信息，从而减少长序列注意力的计算量和显存开销。

本项目包含 BTS 的 PyTorch/CUDA/Triton 实现、Transformers 模型适配，以及
LongBench、RULER、效率和稀疏率评测脚本。评测默认使用 `Qwen/Qwen3-8B`。

## 使用环境

已验证的运行环境如下：

- Linux
- NVIDIA H100 PCIe 80 GB
- Python 3.10
- CUDA 12.8
- PyTorch 2.11.0+cu128
- Triton 3.6.0
- Transformers 5.12.1

项目要求 Python 3.10 及以上版本和支持 BF16 的 NVIDIA GPU。H100（SM90）使用优化路径，
其他 NVIDIA GPU 会使用兼容路径。128K 评测需要较大的显存。

FastDeploy 和 vLLM 吞吐评测需要两个独立环境，不能直接复用上述通用环境：

- FastDeploy：FastDeploy 2.5.0、PaddlePaddle 3.3.1、PaddleFormers 1.1.1、
  Transformers 4.57.6；
- vLLM：PyTorch 2.10.0、FlashAttention 2.8.2、vLLM 0.17.0。

吞吐评测要求独占一张 NVIDIA H100。

## 安装

先下载仓库并进入项目目录：

```bash
git clone https://github.com/Remarkable911/bts.git
cd bts
```

建议创建独立的 Conda 环境：

```bash
conda create -n bts python=3.10 -y
conda activate bts

python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install -r eval/requirements.txt
```

效率评测还会运行 MInference、FlexPrefill 和 XAttention，需要额外安装对比方法依赖：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
python -m pip install packaging psutil ninja
python -m pip install --no-build-isolation \
  -r eval/requirements-baselines.txt
```

### FastDeploy 环境

先按照 [FastDeploy 官方安装说明](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)
安装与 CUDA 匹配的 `fastdeploy-gpu==2.5.0`、`paddlepaddle-gpu==3.3.1` 和
`paddleformers==1.1.1`，再使用同一个 Python 解释器安装本项目：

```bash
export FASTDEPLOY_PYTHON=/absolute/path/to/fastdeploy/bin/python
"$FASTDEPLOY_PYTHON" -m pip install \
  --no-deps --no-build-isolation -e .
```

仅设置 `PYTHONPATH` 不会注册 FastDeploy 工作进程所需的 BTS 插件，因此不能代替上述安装。

### vLLM 环境

vLLM 依赖固定版本的 PyTorch 和 FlashAttention，需按以下顺序安装：

```bash
conda create -n bts-vllm python=3.10 -y
conda activate bts-vllm
export VLLM_PYTHON="$(command -v python)"

"$VLLM_PYTHON" -m pip install "torch==2.10.0" packaging ninja
"$VLLM_PYTHON" -m pip install \
  --no-build-isolation "flash-attn==2.8.2"
"$VLLM_PYTHON" -m pip install "vllm==0.17.0"
"$VLLM_PYTHON" -m pip install -r eval/requirements.txt
"$VLLM_PYTHON" -m pip install \
  --no-deps --no-build-isolation -e .
```

## 评测

所有命令均在项目根目录执行。效率、LongBench、RULER 和稀疏率评测使用通用的 `bts`
环境；FastDeploy 和 vLLM 吞吐评测使用前述两个独立环境。先准备通用评测环境：

```bash
conda activate bts
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN="$(command -v python)"
export RUN_ID="$(date +%Y%m%d-%H%M%S)"
```

模型权重、效率回放数据和 RULER 数据不包含在仓库中。LongBench 可从 Hugging Face
下载默认模型和数据；如使用本地模型，可通过脚本的 `--model` 参数指定路径。

### 效率

该评测在 Qwen3-8B 第 12 层的相同输入上比较 Flash SDPA、MInference、FlexPrefill、
XAttention 和 BTS，默认测试 32K、64K 和 128K 三种长度。

准备 `replay_32768.pt`、`replay_65536.pt` 和 `replay_131072.pt` 后运行：

```bash
export EFFICIENCY_REPLAY_ROOT=/absolute/path/to/efficiency_replays
bash scripts/run_efficiency.sh \
  --output "$PWD/results/efficiency/bts-${RUN_ID}.json"
```

### LongBench

默认运行 21 个任务、共 4,750 个样本，最大长度为 32K：

```bash
bash scripts/run_longbench.sh \
  --output "$PWD/results/longbench/bts-${RUN_ID}"
```

### RULER

先将 `RULER_SUITE_ROOT` 指向准备好的固定 RULER 数据目录。该目录中应包含任务配置、
校验清单和 32K、64K、128K 三种长度的数据；默认模型目录为
`$RULER_SUITE_ROOT/Qwen3-8B-YaRN4`。

```bash
export RULER_SUITE_ROOT=/absolute/path/to/ruler_suite
bash scripts/run_ruler.sh \
  --output "$PWD/results/ruler/bts-selected5-${RUN_ID}"
```

默认评测 5 个任务、3 种长度，共 1,500 个样本。运行完整的 13 个任务时，需指定一个新的
空输出目录：

```bash
bash scripts/run_ruler.sh \
  --tasks ruler \
  --formal \
  --output "$PWD/results/ruler/bts-full-${RUN_ID}"
```

### 稀疏率

该评测使用与 RULER 相同的数据目录，在 128K 输入上统计 BTS 各层的注意力密度。
模型必须已经缓存到本地，也可以通过 `DENSITY_MODEL` 指定本地模型路径。

```bash
export RULER_SUITE_ROOT=/absolute/path/to/ruler_suite
export DENSITY_MODEL=/absolute/path/to/Qwen3-8B
bash scripts/run_density.sh \
  --output "$PWD/results/density/bts-${RUN_ID}"
```

### FastDeploy 与 vLLM 联合吞吐

联合评测使用相同的 128K 合成负载运行 FastDeploy Dense、FastDeploy BTS、vLLM Dense
和 vLLM BTS 四组。需要准备同一 Qwen3-8B YaRN4 权重的 PaddleFormers 与 Hugging Face
模型目录，然后运行：

```bash
export FASTDEPLOY_PYTHON=/absolute/path/to/fastdeploy/bin/python
export VLLM_PYTHON=/absolute/path/to/vllm/bin/python
export FASTDEPLOY_MODEL=/absolute/path/to/Qwen3-8B-YaRN4-paddleformers
export VLLM_MODEL=/absolute/path/to/Qwen3-8B-YaRN4

"$VLLM_PYTHON" -m eval.ThreeWay.execute \
  --fastdeploy-python "$FASTDEPLOY_PYTHON" \
  --vllm-python "$VLLM_PYTHON" \
  --fastdeploy-model "$FASTDEPLOY_MODEL" \
  --vllm-model "$VLLM_MODEL" \
  --bts-config integrations/fastdeploy/configs/topk040_tail256.json \
  --gpu 0 \
  --request-count 16 \
  --wait-for-gpu-seconds 86400 \
  --gpu-idle-stability-seconds 30 \
  --output "$PWD/results/throughput/four-arm-${RUN_ID}"
```

每组进行 5 次独立进程重启；每个请求输入 131,069 个词元并生成 2 个词元。评测输出以下
三项输入词元吞吐加速比及其单侧 95% 置信下界：

- FastDeploy BTS / FastDeploy Dense；
- vLLM BTS / vLLM Dense；
- FastDeploy BTS / vLLM Dense。

输出目录必须不存在。该评测是 128K、单活跃请求、预填充主导的合成负载测试，不评估
生成质量，也不代表多并发在线服务吞吐。完整协议见
[`eval/ThreeWay/README.md`](eval/ThreeWay/README.md)。

上述入口结束后都会在终端打印本次运行的汇总，并将完整结果写入指定的 `results/` 目录。

## 许可证与第三方代码

BTS 原创代码采用 [MIT License](LICENSE)。第三方代码保留各自的许可与版权声明，
具体范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

`baselines/XAttention.py` 和 `baselines/FlexPrefill.py` 中源自 XAttention 的部分
仍有上游许可待确认，不包含在 BTS 的 MIT 授权中。仓库公开和添加 MIT 文件不代表
这些第三方部分已经获得重新授权。
