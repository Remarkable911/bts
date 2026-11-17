# BTS FastDeploy 适配

> **交付边界：**适配器只调用冻结公开入口
> `bts.bts_prefill_attention()`，不导入研发版配置类或内部选择函数。从根项目安装时会
> 同时提供 `bts` 和 `bts_fastdeploy`，并注册 FastDeploy 插件及 `bts-fastdeploy`
> 命令，不需要第二次安装本子项目。源码交付包不附带测试转录、
> GPU 生成输出或验收报告；部署结论必须在目标环境中重新运行本页命令后得出。

Transformers 模型入口仍使用冻结的 attention `forward`。PaddleFormers 模型入口在
QK Norm 和 RoPE 之后把 Q/K/V 交给 FastDeploy；适配器调用相同的冻结张量 API，并在
`rope_already_applied` 标记存在时向 FastDeploy 融合 cache/decode kernel 提供恒等 RoPE，
避免 Q/K 被二次旋转。选择算法不复制，也不修改 FastDeploy 安装目录。

**实现与验证状态要分开：**本适配器实现 FastDeploy 2.5.0 接入，并提供 CPU、源码合同、
插件注册和真实 GPU 生成的独立验证入口。只有目标环境中新生成的报告才能证明该次运行
通过；即使通过，也不自动证明 128K 能力或多请求吞吐。

## 当前范围

| 项目 | 范围 |
|---|---|
| 引擎 | `fastdeploy-gpu==2.5.0`，版本不匹配明确报错 |
| 模型 | Qwen3 文本模型；GPU 验收使用未量化 BF16 Qwen3-8B |
| 设备 | NVIDIA CUDA，单卡，TP=DP=1 |
| 请求 | `1 <= max_num_seqs <= 16`，按 `cu_seqlens` 隔离的完整 causal prefill |
| 算法 | 唯一冻结方法：dual-band mass 0.93、mean128、候选域 block-count Top-K0.4、Tail256 |
| Decode | FastDeploy 原生 attention，保存完整 KV cache |
| Dense 合同 | prefill 固定 FlashAttention 2；decode 固定原生 `append_attention` |
| 暂不支持 | chunked prefill、prefix caching、CUDA Graph、静态图、量化、speculative decoding、分离部署、自定义 mask |
| 实验策略 | Top-P、last-query、Flex、关闭 token sparse、可选 Tail 等非冻结开关均在读取配置时拒绝 |

其他推理引擎还需要各自的适配器。Q/K/V 的共享入口本身不意味着已兼容 vLLM。

## 接入边界

```text
Transformers forward ──→ bts.bts_prefill_attention()
                                  ↑
PaddleFormers Norm/RoPE → FastDeploy cache 写入（恒等 RoPE）→ DLPack → pipeline.py
                                  ↓
                         Paddle attention 输出
```

FastDeploy 的 `FlashAttentionBackend.forward_mixed` 把预处理、cache 写入、prefill、decode 和输出合并放在同一个方法内。`backend.py` 复用该方法的字节码，只在独立 globals 字典中替换 `flash_attn_func`；当上游标记 RoPE 已完成时，临时把其中的 rotary table 换成 `cos=1、sin=0`，调用结束后恢复元数据。其余 cache、decode 和输出合并步骤调用发行包原实现。不修改 FastDeploy 安装文件，不覆盖进程级 `flash_attn_func`，不触碰 Transformers 类。

`bridge.py` 按 FastDeploy 的 `cu_seqlens` 拆分最多 16 个 ragged 请求，逐段调用冻结入口并
按原 packed 顺序写回；空 slot 会被跳过。mixed iteration 中只处理
`seq_lens_encoder > 0` 的完整 prefill，已有请求的 decode 仍由 FastDeploy 原生
append-attention 计算和 merge。`pipeline.py` 校验每段完整 prefill 的 Q/K/V 边界，然后
调用冻结公开入口；padding、选择和 kernel dispatch 均由 `bts` 核心负责。DLPack 共享
Q/K/V 存储，Torch 运算在 Paddle 当前 CUDA stream 上执行，DLPack 管理输出的存储
生命周期。这里只优化 attention 计算，没有删除 KV cache。

## 环境与安装

建议在独立部署环境中安装 FastDeploy、Paddle 和本提交，以免改变已有实验环境。

目标 FastDeploy 版本为 2.5.0，Paddle 为 3.3.1。官方 CUDA 轮子来自
[FastDeploy 安装说明](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)，
不是 PyPI 上其他同名项目。FastDeploy 2.5 要求 Transformers `>=4.55.1,<5`；根项目
支持 `>=4.55.1,<6`，并通过 `fastdeploy` extra 将部署环境显式收窄到 `<5`。因此根项目
可在 FastDeploy 2.5 环境中一次安装冻结核心和适配器，不再存在依赖区间冲突。

本包不附带冻结环境报告。部署者需记录实际 Torch、Triton、Paddle、FastDeploy、
PaddleFormers、Transformers、CUDA 与 GPU 版本，并在该组合上重新执行下述检查。

PaddleFormers 1.1.1 会安装提供 `torchcodec` 模块但不提供同名 distribution 元数据的
`paddlecodec`；这与 Transformers 4.57.6 的音频可选依赖检测冲突。本适配仅处理文本；
部署时应单独评估该可选解码包，不要据此配置部署音频/视频模型。

先按官方说明安装匹配硬件的 FastDeploy 2.5/Paddle 轮子。它们没有列为本项目的自动
依赖，因为发行地址和 CUDA 变体依赖部署环境。随后从提交根目录执行一次安装：

```bash
python -m pip install -e '.[fastdeploy]'
```

该根目录安装同时提供两个包并注册可选插件，但必须设置
`BTS_FASTDEPLOY_ENABLE=1` 才会修改 FastDeploy 的 backend 选择；普通
`import bts`、`import bts_fastdeploy` 和未启用的 FastDeploy 都不会激活适配。
必须使用实际启动 FastDeploy 的同一个 Python 环境完成安装；仅设置 `PYTHONPATH`
不会创建 GPU worker 依赖的 distribution entry point。启动器会在占用 GPU 和加载权重
之前检查 `bts -> bts_fastdeploy.plugin:register`，缺失或指向错误时立即失败。

`integrations/fastdeploy/pyproject.toml` 仅供单独开发适配器时使用，并声明已安装的
`bts[fastdeploy]==0.3.0` 为依赖；完整提交的部署不需要执行该目录的
第二次安装。

## 独立运行

首先检查配置，不加载 Paddle/FastDeploy，也不运行 GPU：

```bash
python -m bts_fastdeploy.run \
  --model /path/to/Qwen3-8B \
  --config integrations/fastdeploy/configs/topk040_tail256.json \
  --output /path/to/output/dry-run --dry-run
```

在空闲 GPU 上执行生成：

```bash
CUDA_VISIBLE_DEVICES=0 python -m bts_fastdeploy.run \
  --model /path/to/Qwen3-8B \
  --prompt-file /path/to/prompt.txt \
  --config integrations/fastdeploy/configs/topk040_tail256.json \
  --output /path/to/output/generation
```

验证 16 个请求的 packed prefill（重复 prompt 只用于接入 smoke，不作为吞吐报告）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m bts_fastdeploy.run \
  --model /path/to/Qwen3-8B \
  --max-num-seqs 16 --num-prompts 16 \
  --max-model-len 2048 --max-new-tokens 2 \
  --output /path/to/output/batch16-smoke
```

省略 `--config` 使用唯一冻结配置。示例文件只重述其中的 Top-K0.4 和 Tail256 字段；
传入值必须与冻结方法完全一致。配置读取器会拒绝 Top-P、last-query、Flex、FP16 以及
其他会改变算法或数值语义的研发选项，也不读取当前 shell 的算法环境变量覆盖配置。

启动器设置 `FD_ATTENTION_BACKEND=FLASH_ATTN` 和插件白名单，将配置与源码路径传入 worker。只在 FastDeploy 进程内启用适配。`generation_report.json` 检查真实生成期间的新增 BTS 调用精确覆盖模型全部层，并保存配置哈希、各层调用次数和生成 token；预热调用不算真实请求证据。审计写盘会影响计时，这些报告用于正确性验收，不是效率报告。

FastDeploy 2.5 在 CUDA mixed 模式下会把普通参数
`enable_chunked_prefill=False` 自动改回开启；该版本提供的强制关闭入口是环境变量
`FD_DISABLE_CHUNKED_PREFILL=1`。本启动器会在任何 FastDeploy 导入和 worker 创建之前
强制设置并校验该值，且把它连同其他引擎环境约束写入 dry-run 和生成报告。标准插件
加载路径在变量未设置时也会补为 `1`；如果调用方显式给出其他值则立即拒绝，避免以
不受支持的 chunked-prefill 语义继续执行。

## 验证

```bash
# CPU：含冻结公开入口对齐、恒等 RoPE 防二次旋转、Paddle 实际桥接、未启用时无副作用
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q integrations/fastdeploy/tests

# 检查安装的 FastDeploy 发行包源码/方法签名，不启动 GPU
python integrations/fastdeploy/scripts/check_source_contract.py

# 验证真实插件与 selector；只替代 SM 元数据以允许隐藏 GPU 的 import 检查
CUDA_VISIBLE_DEVICES='' python integrations/fastdeploy/scripts/check_registration.py

# 等 GPU 连续空闲 120 秒再运行 GPU 桥接和两种配置的真实生成
python integrations/fastdeploy/scripts/validate_gpu.py \
  --model /path/to/Qwen3-8B --output /path/to/acceptance \
  --gpu-index 1 --cooperative-lease-dir /path/to/gpu-leases \
  --gpu-memory-utilization 0.45 --wait
```

冻结 BTS kernel 要求 `head_dim=128`。典型 Qwen3-8B 验收模型配置为
`hidden_size=4096`、32 个 query heads、8 个 KV heads、36 层，显式
`head_dim=128`，权重为未量化 BF16。等待程序会在
查询或等待 GPU 之前读取模型 `config.json`，要求 `model_type=qwen3`、未量化，并优先
使用显式 `head_dim`；只有该字段缺失时才由 `hidden_size / num_attention_heads` 推导。
结果不是 128 的模型会立即退出，不会占用 GPU。`--gpu-memory-utilization` 允许范围为
`(0, 1]`，默认值为 `0.20`，可按空闲设备容量显式调整。

`--gpu-index` 使用 `nvidia-smi` 显示的物理编号，可重复传入以形成允许列表；例如仅传
`--gpu-index 1` 时，GPU 0 即使先空闲也不会被本次验收使用。不传时仍会在所有可见
设备中选择。`--cooperative-lease-dir` 是可选的现有目录，供本机其他队列也采用
`gpu{物理编号}.lock` 约定时使用；等待器必须非阻塞取得对应文件锁并在整个验收期间
保持，锁被其他队列占用时会继续等待。目录由部署者在命令行指定，脚本不硬编码实验
仓库路径。

等待程序只有在 GPU 无计算进程、显存使用低于 1 GiB、利用率不超过 5% 时才开始计时；不会停止已有进程。默认最多等 12 小时，可通过 `--timeout-hours` 调整。`status.json` 区分等待、超时、失败、中断和通过，逐步日志保留在同一目录。每个验证步骤运行在独立进程组；步骤非零退出、超时或主验证器收到 SIGINT/SIGTERM 时，会先终止、再强制清理该步骤的进程组，避免 FastDeploy worker 在验证器退出后继续占用 GPU。中断状态记录信号并返回常规的 `128 + signal` 退出码。

它不是系统级 GPU 调度器，不能阻止不采用同一租约约定的其他用户在最终检查之后同时提交任务；`--cooperative-lease-dir` 只与共同使用 `gpu{index}.lock` 的本机队列协调。

研究仍在修改代码时，可先复制 `bts/` 源码并记录哈希，通过 `--bts-source /path/to/snapshot` 指定验证快照；这样等待期间的研究修改不会混入本次验收。验证使用单独的 Torch extension/Triton cache。

验收最终交付时，应从最终源码压缩包解出的独立目录运行验证，并让
`--bts-source` 指向该目录。这样核心源码和适配器都来自同一份交付快照，避免验收混入
仍在修改的研究工作树。

等待阶段只通过 `nvidia-smi` 观察设备，不导入 Torch、Paddle 或 FastDeploy。选中空闲
GPU 后，脚本从当前 Python 的 `sys.path` 查找 `torch/lib` 与 `nvidia/*/lib`，连同已有
`LD_LIBRARY_PATH` 一起传给验证子进程，避免桥接预检在加载 Paddle 时因找不到
`libcudnn.so.9` 提前失败；这个只读路径发现过程本身不会创建 CUDA context。

GPU 测试默认跳过，只有设置 `BTS_FASTDEPLOY_TEST_CUDA=1` 才启用。等待程序会为选中的 GPU 设置这个开关。

最终源码压缩包必须包含根项目安装配置、`bts/`、本目录的适配器源码、配置、脚本和
测试，并排除运行结果、缓存及构建产物。应从最终解压目录重新执行本页的 CPU、源码合同、
插件注册和 GPU 验收，不能复用开发目录中的历史输出。

## 后续研究和扩展

冻结核心更新后，需要重新做固定 Q/K/V 对齐和 FastDeploy 验收，再发布新的部署快照；
适配层始终复用公开张量入口，不另写一套 Paddle 选择算法。

后续分页 KV 原生读取、连续批处理和 chunked prefill 都在适配层内迭代。尤其 tail256 必须按完整 prompt 的位置解释，不能按每个 chunk 的末尾重新保护。
