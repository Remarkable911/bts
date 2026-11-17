# 评测指南

所有命令均在仓库根目录执行。先按 [主 README](../README.md#安装) 安装 `bts` 环境及
评测依赖，再设置 GPU 和本次运行标识：

```bash
conda activate bts
export CUDA_VISIBLE_DEVICES=0
export RUN_ID="$(date +%Y%m%d-%H%M%S)"
```

模型权重、Efficiency 回放输入和固定 RULER 数据套件不随仓库发布，需提前准备。
下方路径按实际位置替换，每次运行使用新的输出路径。

## Efficiency

比较 Qwen3-8B 第 12 层在 32K、64K、128K 下的注意力耗时。
需安装 `eval/requirements-baselines.txt` 中的对比方法依赖，输入目录需包含
`replay_32768.pt`、`replay_65536.pt` 和 `replay_131072.pt`。

```bash
export EFFICIENCY_REPLAY_ROOT=/absolute/path/to/efficiency_replays
bash scripts/run_efficiency.sh --output "results/efficiency/bts-${RUN_ID}.json"
```

切换层或输入方式见 [Efficiency 说明](efficiency/README.md)。

## LongBench

默认使用 Qwen3-8B，评测 21 个任务、4,750 个样本，最大长度为 32K。
模型和数据可从 Hugging Face 下载，本地模型可用 `--model` 指定。

```bash
bash scripts/run_longbench.sh --output "results/longbench/bts-${RUN_ID}"
```

## RULER

准备包含任务配置、校验清单和 32K/64K/128K 数据的固定 RULER 套件；默认模型位于
`$RULER_SUITE_ROOT/Qwen3-8B-YaRN4`。默认运行 5 个任务，共 1,500 个样本。

```bash
export RULER_SUITE_ROOT=/absolute/path/to/ruler_suite
bash scripts/run_ruler.sh --output "results/ruler/bts-${RUN_ID}"
```

完整 13 个任务加上 `--tasks ruler --formal`。数据目录要求见 [RULER 说明](RULER/README.md)。

## Density（稀疏率）

使用固定 RULER 套件及其官方输出日志，在 128K 输入上统计 BTS 的注意力密度。
模型使用已下载到本地的基础 Qwen3-8B。

```bash
export RULER_SUITE_ROOT=/absolute/path/to/ruler_suite
export DENSITY_MODEL=/absolute/path/to/Qwen3-8B
bash scripts/run_density.sh --output "results/density/bts-${RUN_ID}"
```

输入与统计口径见 [Density 说明](Density/README.md)。

## 吞吐评测

FastDeploy 和 vLLM 分别使用独立环境，并在各自环境中安装本项目。版本与安装步骤：

| 环境 | 主要版本 | 安装说明 |
| --- | --- | --- |
| FastDeploy | FastDeploy 2.5.0、PaddlePaddle 3.3.1、PaddleFormers 1.1.1、Transformers 4.57.6 | [安装](../integrations/fastdeploy/README.md#环境与安装) |
| vLLM | PyTorch 2.10.0、FlashAttention 2.8.2、vLLM 0.17.0 | [安装](VLLM/README.md#1-安装) |

准备同一 Qwen3-8B YaRN4 权重的两种模型目录，并使用一张独占的 H100：

```bash
export FASTDEPLOY_PYTHON=/absolute/path/to/fastdeploy/bin/python
export VLLM_PYTHON=/absolute/path/to/vllm/bin/python
export FASTDEPLOY_MODEL=/absolute/path/to/Qwen3-8B-YaRN4-paddleformers
export VLLM_MODEL=/absolute/path/to/Qwen3-8B-YaRN4
```

### 四臂联合吞吐

在 128K 合成输入上比较两个引擎各自的 Dense 和 BTS，输出吞吐及加速比。
这是预填充为主的测试，模型与 tokenizer 内容需符合脚本固定的哈希校验。

```bash
bash eval/ThreeWay/reproduce_128k.sh "results/throughput/four-arm-${RUN_ID}" 0
```

参数与测量范围见 [四臂评测说明](ThreeWay/README.md)。

### 单引擎实验（开发用途）

FastDeploy 内部的 Dense/BTS 对比：

```bash
"$FASTDEPLOY_PYTHON" -m eval.FastDeploy.run \
  --model "$FASTDEPLOY_MODEL" --output "results/fastdeploy/bts-${RUN_ID}"
```

vLLM 先生成 RULER 请求，再运行 Dense/BTS 对比：

```bash
"$VLLM_PYTHON" -m eval.VLLM.workload \
  --model "$VLLM_MODEL" --output "results/vllm/workload-${RUN_ID}"
"$VLLM_PYTHON" -m eval.VLLM.run \
  --model "$VLLM_MODEL" --workload "results/vllm/workload-${RUN_ID}" \
  --output "results/vllm/bts-${RUN_ID}"
```

其他参数见 [FastDeploy](FastDeploy/README.md) 和 [vLLM](VLLM/README.md) 说明。
