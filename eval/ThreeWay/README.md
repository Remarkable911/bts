# 128K 四臂吞吐实验

> **发布范围：** 本仓库提供四臂实验协议及
> [`reproduce_128k.sh`](./reproduce_128k.sh) 入口。开发阶段的实验报告和原始测量产物
> 未随本仓库发布；本页不提供可供独立核验的历史吞吐结果。
> 请配置自己的运行环境和模型路径，并以新运行生成的报告为准。

[`run.py`](./run.py) 只负责**合同校验和统计分析**；[`execute.py`](./execute.py)
是对应的受信执行层，会用两个独立 Python 环境实际启动 FastDeploy/vLLM 四臂。
四个 arm 最终各自产出一个标准化 report：

| arm | prefill | decode |
|---|---|---|
| `fd_dense` | FastDeploy FlashAttention 2 | FastDeploy 原生 append-attention |
| `fd_bts` | FastDeploy BTS | FastDeploy 原生 append-attention |
| `vllm_dense` | vLLM FlashAttention 2 | vLLM 原生 FlashAttention 2 |
| `vllm_bts` | vLLM BTS | vLLM 原生 FlashAttention 2 |

分析器计算且只计算以下三个方向的加速比：

```text
fd_bts / fd_dense
vllm_bts / vllm_dense
fd_bts / vllm_dense
```

这里的除法是 input-token throughput 之比。因为四臂的请求数和每请求输入 token 数完全
相同，它也等于相应 arm 的 `measured_seconds` 反比。

## 冻结的、偏向 prefill 收益的配置

- 128K context budget：`max_model_len=max_num_batched_tokens=131072`；
- 每请求 **131069 input tokens + 2 output tokens**；
- `max_num_seqs=1`，无限 offered load，共 16 条完全相同 workload 合同的请求；128K
  prefill 在 `max_num_batched_tokens=131072` 下本来就无法同时批入两条，这个设置还使
  FastDeploy profile 使用一条接近真实长度的 128K dummy，而不是 16 条 8K dummy，
  从而捕获 BTS 的真实长序列峰值并避免过量分配 KV cache；
- BF16 model/KV cache，TP=PP=DP=1，`gpu_memory_utilization=0.90`；
- eager execution、关闭 prefix caching 和 chunked prefill；
- greedy sampling（`temperature=0`），强制生成 2 token 并在这两个 token 内有效忽略
  EOS，防止某一臂提前结束而没有执行真实 decode。vLLM 直接设置
  `ignore_eos=true`（raw-token 模式同时要求 `detokenize=false`）；FastDeploy 2.5 的
  `SamplingParams` 没有该字段，因此执行器设置
  `min_tokens=max_tokens=2`，在本合同的固定输出长度内得到等价的 EOS 屏蔽语义；
- 两个引擎的 dense prefill 都明确选择并审计 FlashAttention 2；BTS 的 dense fallback
  策略冻结为 `engine_native_dense`：FastDeploy decode 使用原生 append-attention，vLLM
  decode 使用原生 vLLM FlashAttention 2；
- BTS 只允许优化完整 prefill，decode 中 BTS 调用数必须为 0；
- 启动和 warmup 不计时，每个 arm 做 5 个 fresh-process restart，并要求 GPU 独占。

输出不是 1 token：只生成 1 token 时通常只有 prefill/sampling，没有实际 decode
迭代，无法证明“BTS 只优化 prefill、decode 走最佳 dense 路径”。2-token 输出只增加
一个 decode step，仍然最大化 128K prefill 在总时间中的占比，同时能用审计计数证明
真实 dense decode。FastDeploy 还要求在 prompt 与生成 token 之外保留一个边界位置，
因此共同 prompt 长度是 `131072 - 2 - 1 = 131069`。

## Arm report 合同

输入 schema 为 `bts_four_arm_input_v1`。顶层必须且只能包含：

```json
{
  "schema_version": "bts_four_arm_input_v1",
  "arm": "fd_bts",
  "shared_contract": {},
  "shared_contract_sha256": "...",
  "workload": {},
  "workload_sha256": "...",
  "runtime": {},
  "restarts": []
}
```

完整字段和固定值由 [`run.py`](./run.py) 中的 `_SHARED_CONTRACT_KEYS`、
`_WORKLOAD_KEYS`、`_RUNTIME_KEYS` 和 `_RESTART_KEYS` 定义。对象哈希是
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` 的
UTF-8 SHA-256。worker 可以直接调用公开函数 `build_shared_contract()`、
`build_workload_contract()` 和 `build_arm_report()` 生成并自校验 schema，或单独调用
`canonical_sha256()`。

四臂的 `shared_contract` 和 `workload` 必须逐字段完全相同，且各自声明的哈希必须与
对象重算值一致。`logical_model_sha256` 是跨 HF/Paddle 导出共享的、规范化逻辑模型
身份，不是两个格式必然不同的原始 `config.json` 文件哈希。workload 同时冻结有序
request ID 和有序 prompt token IDs 的哈希，从而禁止四臂使用不同 prompt、tokenizer
结果或请求顺序。同一引擎的 dense/BTS 两臂还必须具有完全相同的 `engine_version`、
`engine_config_sha256` 和 `source_manifest_sha256`；attention mode 是两臂之间唯一允许的
runtime 路由差异。

`runtime.observed_dense_attention_backend` 必须来自 prefill 运行时 backend 选择审计，
不能只回显命令行参数。`runtime.decode_backend` 则必须精确匹配上述引擎原生 dense
decode 路径，不能把 FastDeploy append-attention 错报为 FA2。每个 restart 还必须报告
prefill/decode 的 dense/BTS 调用计数；BTS arm
要求 BTS prefill 非零、dense prefill 为零、dense decode 非零、BTS decode 为零。
Dense arm 则要求 dense prefill/decode 均非零且 BTS 计数为零。

分析器只能校验报告内部证据的一致性，不能从 JSON 本身证明 worker 没有伪造
`observed_*`、GPU 独占或调用计数。正式结论只接受由受信的冻结 worker 直接产生、并由
外部产物保全其日志和源码清单的 arm report；手工编辑的 JSON 只能用于测试分析流程。

正式执行器不会把配置中的期望值冒充运行时证据。FastDeploy 的 FA2 prefill 和原生
append-attention decode 合同同时由 launcher 和 GPU-worker audit 验证；由于 FastDeploy
可能把下一条 prefill 与上一条 decode 混批，旧的 `native_decode_calls` 会漏计，执行器
改用“每条请求实际返回恰好 2 token + worker 已观测 `decode_op=append_attention`”作为
decode 证据。vLLM BTS 的 prefill/decode 分别要求 mmap 中的 `sparse_segments` 和
`dense_fallback_segments` 精确等于 `层数 × 请求数`；两个 vLLM arm 还必须在 model
worker 日志中实际报告 FlashAttention version 2。vLLM 0.17 wheel 自带的 FA2 CUDA
binary 不作为可回退路径：仓库插件把其 22 参数 Python/op 合同严格映射到固定的
`flash-attn==2.8.2` upstream 21 参数 ABI，两个 vLLM arm 都显式使用 256-token KV block。
日志还必须包含 adapter 的首次真实调用 marker（版本、21 参数和 page size=256）；
marker 会记录实际导入的扩展路径、SHA-256 及其对 `flash-attn` distribution
文件清单的归属。每个 model-worker 还在 mmap 中维护累计 adapter 调用计数；
执行器分别在 warmup 后与计时 `generate` 返回后读取，只接受对应请求区间的精确
差值，并要求 marker PID 与该计时 worker PID 一致。因此仅证明 warmup、安装插件
或回显配置都不足以成为运行证据。缺 upstream 扩展、binary 归属/哈希、版本/
ABI 或 page size 不兼容都会 fail closed。

`engine_configs` 表示冻结的**有效配置合同**，不表示其中每个字段都作为同名构造参数
传入。FastDeploy worker 在 `LLM` 创建后读取 `llm.llm_engine.cfg`，把 model BF16、KV
BF16、TP/DP、序列/批 token 上限、0.90 显存比例、关闭 prefix/chunk、graph level 0/
无 CUDA graph 等 resolved 值及逐字段来源写入 observation，并逐项对照合同后才 warmup。
FastDeploy 2.5 没有 pipeline-parallel EngineArgs/Config 字段；执行器把 TP=DP=1 的单模型
阶段能力约束明确记录为 PP=1 的来源，而不是声称传入了一个不存在的 PP 参数。采样证据
也会分别记录 vLLM 的显式 `ignore_eos=true` 与 FastDeploy 的上述等价实现。

## 统计和门槛

统计单位是同编号的 fresh-process restart pair。每个比值先计算 5 个 paired restart
ratio，再取中位数。分析器用 seed `20261110` 做 10,000 次 paired-cluster percentile
bootstrap；每次重采样完整 restart index，而且三个比较共享同一组抽样 index。

正式硬门槛是：三个比较在 128K 下的单侧 95% bootstrap lower bound 都
`>= 1.5`。点估计超过 1.5 但 lower bound 不足仍是 `FAIL`。

## 实际执行四臂

输出目录必须尚不存在。FastDeploy 和 vLLM 可以安装在不同 venv；执行器用指定解释器
分别启动 worker，不要求在控制器环境同时安装两个引擎：

先在仓库根目录配置两个环境和对应的模型目录，再执行复现脚本：

```bash
export FASTDEPLOY_PYTHON=/absolute/path/to/fastdeploy/bin/python
export VLLM_PYTHON=/absolute/path/to/vllm/bin/python
export FASTDEPLOY_MODEL=/absolute/path/to/Qwen3-8B-YaRN4-paddleformers
export VLLM_MODEL=/absolute/path/to/Qwen3-8B-YaRN4
WAIT_FOR_GPU_SECONDS=604800 bash eval/ThreeWay/reproduce_128k.sh
```

脚本要求显式提供上述四个路径；缺失时会在加载模型前报错。默认输出写入仓库内新的
`results/throughput/four-arm-时间戳` 目录，也可通过第一个位置参数或 `OUTPUT` 指定。
该入口保留参考实验的模型与 tokenizer 内容哈希，迁移目录不改变这些校验条件。

脚本结束前会打印四臂测量耗时中位数、吞吐中位数、三项加速比、单侧 95% 下界、
`1.5x` 目标、整套控制器墙钟耗时及结果路径，不显示 PASS/FAIL 状态文字。以下是需要
自行安装环境或覆盖模型路径时的底层配置方式：

正式执行期间，控制器会在终端持续刷新一个总计 `20` 个 logical worker 的 tqdm
进度条，显示当前 restart、arm、attempt、阶段、当前步骤耗时、总耗时和 ETA。失败重试
只更新 attempt/阶段，不会虚增完成数；进度信息写入 stderr，最终 JSON stdout 保持不变。

```bash
VLLM_PYTHON=/venvs/vllm/bin/python

"$VLLM_PYTHON" -m pip install "torch==2.10.0" packaging ninja
"$VLLM_PYTHON" -m pip install \
  --no-build-isolation "flash-attn==2.8.2"
"$VLLM_PYTHON" -m pip install "vllm==0.17.0"
"$VLLM_PYTHON" -m pip install -r eval/requirements.txt
"$VLLM_PYTHON" -m pip install --no-deps -e .
```

`.[vllm]` 只锁定 vLLM，故意不包含需要上述分阶段构建的 `flash-attn`。确认
该 venv 后再运行四臂执行器：

```bash
python -m eval.ThreeWay.execute \
  --fastdeploy-python /venvs/fastdeploy/bin/python \
  --vllm-python /venvs/vllm/bin/python \
  --fastdeploy-model /models/Qwen3-8B-YaRN4-fastdeploy \
  --vllm-model /models/Qwen3-8B-YaRN4-vllm \
  --bts-config integrations/fastdeploy/configs/topk040_tail256.json \
  --gpu 0 \
  --wait-for-gpu-seconds 86400 \
  --gpu-idle-stability-seconds 30 \
  --output /results/three-way-128k
```

`--gpu` 可以是物理 index 或完整/唯一前缀 GPU UUID；控制器记录 UUID 后统一给两个引擎
传物理 index，以规避 FastDeploy 2.5 对 UUID selector 的整数转换问题。执行前会逐文件
校验两模型目录中相同的 5 个 safetensors shard、index 和 tokenizer 文件，并结合规范化
关键 config 自动生成 `logical_model_sha256` 与 `tokenizer_sha256`。可选的
`--logical-model-sha256`/`--tokenizer-sha256` 只作为额外期望值，不能覆盖实测文件哈希。

`--wait-for-gpu-seconds` 允许控制器等待指定 GPU 真正空闲。一旦抢占成功，控制器会先用
vLLM Python 启动一个持有 1-byte tensor 的空闲 CUDA lease；该 lease PID 从第一个 arm
之前一直保留到 20 个 arm、最终分析和结果落盘全部结束，因此模型卸载和 fresh worker
切换期间 GPU 也不会重新显示为空闲。lease 不发计算 kernel，只保留最小 CUDA context，
四臂受到的显存条件完全相同；每个模型 worker 仍是独立 fresh process。每臂结束时 GPU
状态被如实审计为 `lease_only`，不是虚报为 idle。CUDA lease 能让普通调度器看到 GPU
已占用，但不能阻止其他用户强行在同一 GPU 上启动进程。若出现任何外部 PID，当前
attempt 会立即停止且永久标记为不可用于统计；控制器只清理自己带 run-token 的进程，
绝不会终止对方进程，也不会释放 CUDA lease。控制器在 lease 持续占用 GPU 的状态下等待
外部 PID 离开并重新连续稳定 30 秒，然后用新的 run-token 重试同一 logical arm。默认
污染重试次数和恢复等待时间均不设上限；可用
`--max-contamination-retries-per-arm` 与 `--gpu-recovery-timeout-seconds` 设置上限。
worker crash、OOM、非零退出或 timeout 也会先完成 identity-verified cleanup，在 lease
持续存在的状态下重新确认 `lease_only` 稳定窗口，再以 fresh process 重试同一 arm；它
使用独立于污染次数的预算，默认同样不设上限，可用
`--max-worker-failure-retries-per-arm` 限制。配置/协议/observation 校验错误不重试，避免
把确定性错误伪装成偶发 worker 故障。
默认还要求 GPU 连续空闲 30 秒后才建立 lease，避免在另一个批处理作业轮换子进程时把
短暂的进程间隙误判成作业已经结束；可用 `--gpu-idle-stability-seconds` 显式调整。

正式控制器还会从等待 GPU 开始一直持有仓库级测试锁和项目现有的三个 GPU 协作锁，直到
lease 最终释放。仓库根目录和 FastDeploy 测试目录的 `conftest.py` 共用同一 guard，使
普通 pytest 在导入测试模块前默认设置
`CUDA_VISIBLE_DEVICES=""`；确实需要 CUDA 的测试必须显式设置
`BTS_REPO_TEST_CUDA=1 CUDA_VISIBLE_DEVICES=<单个物理 GPU>`，并成功取得同一组锁，否则
pytest 会在加载测试模块前拒绝启动。

每个 restart 的 arm 顺序循环轮转；每个成功 arm 都来自 fresh subprocess。控制器每秒
检查 GPU compute PID，只允许 lease、当前 worker 及其带 run-token 的子进程。外部进程
触发上述保留 lease 的恢复流程；worker 残留、源码变化、模型变化或审计缺失仍会 fail
closed。若引擎 GPU 子进程已经消失但 launcher 未退出，控制器会在 60 秒后报告可能的
crash/OOM，而不是一直挂到四小时 worker timeout。输出目录包含：

- `controller-spec.json`、源码/模型身份和命令计划；
- `gpu-lease.json`：持续 lease PID、最小显存标记、获取/释放时间和释放后真实 GPU 状态；
- 每个 `restart-N/arm/attempt-NNN/` 的命令、完整日志、原始 observation、GPU 独占记录
  和审计证据；受污染或 worker 失败的 attempt 与随后恢复过程会被完整保留，但不会进入
  arm report；
- 四份标准 arm report：`fd_dense.json`、`fd_bts.json`、`vllm_dense.json`、
  `vllm_bts.json`；
- 自动生成的最终 `three-comparison.json`。

无 GPU 环境可以先检查全部命令和静态合同；dry-run 不检查文件、不导入推理引擎、
不查询 NVIDIA 状态，也绝不会生成测量 report：

```bash
python -m eval.ThreeWay.execute [同上参数] --dry-run
```

## 只分析已有 arm report

```bash
python -m eval.ThreeWay.run \
  --fd-dense /results/fd-dense.json \
  --fd-bts /results/fd-bts.json \
  --vllm-dense /results/vllm-dense.json \
  --vllm-bts /results/vllm-bts.json \
  --output /results/three-comparison.json
```

分析输出文件和实际执行的输出目录都拒绝覆盖已有目标。两个入口的退出码均为：

- `0`：三个 lower bound 全部通过；
- `2`：合同有效，但至少一个比较没有证明 1.5x；
- `4`：输入、共享合同、workload、审计或输出文件不符合合同。

