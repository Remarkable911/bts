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

## 评测

在项目根目录执行，使用 `CUDA_VISIBLE_DEVICES` 选择 GPU：

```bash
conda activate bts
export CUDA_VISIBLE_DEVICES=0
```

### Efficiency

在 Qwen3-8B 第 12 层比较 BTS 与对比方法的注意力耗时，默认测试 32K、64K、128K。
需自行准备包含 `replay_32768.pt`、`replay_65536.pt`、`replay_131072.pt` 的输入目录：

```bash
export EFFICIENCY_REPLAY_ROOT=/absolute/path/to/efficiency_replays
bash scripts/run_efficiency.sh
```

### LongBench

使用 Qwen3-8B 运行 21 个任务、共 4,750 个样本，最大长度为 32K。
首次运行可从 Hugging Face 下载模型和数据：

```bash
bash scripts/run_longbench.sh
```

结果默认保存到 `results/`；重复运行可用 `--output` 指定新的输出路径。
各项评测的用法，以及 RULER、稀疏率和吞吐评测入口，见 [评测指南](eval/README.md)。

## 许可证与第三方代码

BTS 原创代码采用 [MIT License](LICENSE)。第三方代码保留各自的许可与版权声明，
具体范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

`baselines/XAttention.py` 和 `baselines/FlexPrefill.py` 中源自 XAttention 的部分
仍有上游许可待确认，不包含在 BTS 的 MIT 授权中。仓库公开和添加 MIT 文件不代表
这些第三方部分已经获得重新授权。
