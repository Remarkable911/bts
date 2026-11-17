# vLLM 0.17：RULER serving 吞吐实验（dev-only）

> **开发态说明：** 当前没有完整 formal 配对结果或可作为提交“最优结果”的证据，
> 对应的稳定 `scripts/` 入口也未随提交保留。以下内容仅是开发协议记录，不应被当作
> 已完成的提交评测或结果声明。

本评测在同一个 `vllm==0.17.0` 引擎和 stock scheduler 中比较两臂：

- `dense`：vLLM 原生 `FLASH_ATTN` attention backend；
- `bts`：通过 `AttentionBackendEnum.CUSTOM` 注册的 BTS backend。完整、无缓存的
  prefill 使用 BTS，decode 和带历史 KV 的 extension 保留 vLLM 原生 FlashAttention。

两臂使用相同的模型、RULER 请求、KV cache、调度参数和输出 token budget。runner 为每个
配对 restart 启动全新的服务进程并交替两臂顺序；BTS 臂还必须通过逐层稀疏调用审计。
这比跨引擎比较更容易解释，但结论范围仍是“BTS sparse-attention system 在 stock vLLM
scheduler 内相对 dense vLLM 的吞吐”，不是“新调度模块本身”的单独归因。
当前运行合同为 `vllm_dense_vs_bts_ruler_serving_fa2_v3`。dense 与 BTS 两臂共同
固定 `flash_attn_version=2` 并使用 eager execution；BTS 只替换完整 prefill，decode
与 extension 仍走父类原生 FlashAttention 2。这保证两臂的 attention 差异只来自
prefill BTS，而不是 FA2/FA3 或 CUDA Graph 配置差异。

## 1. 安装

backend 使用 vLLM 的私有 attention API，因此版本被严格锁定为 `0.17.0`。
两臂还依赖单独构建的 upstream `flash-attn==2.8.2`。该包在 setup 阶段就导入
Torch，所以必须在运行 server 的同一个 Python 解释器中按下列顺序安装：

```bash
cd /path/to/prism-token-sparse-submission
PYTHON_BIN=/path/to/python

"$PYTHON_BIN" -m pip install "torch==2.10.0" packaging ninja
"$PYTHON_BIN" -m pip install \
  --no-build-isolation "flash-attn==2.8.2"
"$PYTHON_BIN" -m pip install "vllm==0.17.0"
"$PYTHON_BIN" -m pip install -r eval/requirements.txt
"$PYTHON_BIN" -m pip install \
  --no-deps --no-build-isolation -e "$PWD"

"$PYTHON_BIN" -c \
  'import importlib.metadata as m; print(m.version("torch"), m.version("flash-attn"), m.version("vllm")); print([(e.name, e.value) for e in m.entry_points(group="vllm.general_plugins") if e.name in {"bts_vllm", "bts_vllm_fa2_compat"}])'
```

最后一条命令应打印 `2.10.0 2.8.2 0.17.0` 与两个提交内 entry
point。`.[vllm]` 故意不包含 `flash-attn`：pip extra 不能表达“先安装 Torch，
再以 `--no-build-isolation` 构建 FA2”的必要顺序。只设置 `PYTHONPATH` 也不会
注册这两个入口；runner 会在加载模型和等待 GPU 之前拒绝错误版本或缺失入口。
运行时会进一步确认实际导入的 `flash_attn_2_cuda` 是 `flash-attn`
distribution 清单中的本机扩展，并在证据中记录解析路径与 SHA-256。正式模型
合同是本地 Qwen3-8B YaRN4 长上下文配置：36 层、32 个 query heads、
8 个 KV heads、head dimension 128、BF16，并能覆盖 131072-token context。

## 2. 导出冻结的实际任务 workload

`workload.py` 使用提交内置的 13 个 RULER task builder 和模型 tokenizer，确定性构造
32K、64K、128K 请求；这一步不加载模型权重。默认每个 task 导出 8 条请求，即每个长度
104 条。prompt 已应用 chat template 且关闭 Qwen3 thinking，每条记录保留 task、token
数和内容哈希。

```bash
MODEL=/path/to/Qwen3-8B-YaRN4
WORKLOAD=/external/results/vllm-ruler-workload

"$PYTHON_BIN" -m eval.VLLM.workload \
  --model "$MODEL" \
  --output "$WORKLOAD" \
  --lengths 32768,65536,131072 \
  --samples-per-task 8
```

输出目录必须是新目录或空目录。它包含 `manifest.json` 和三个
`ruler-<length>.jsonl`；runner 会复核 manifest、文件 SHA-256、prompt token 数及模型
config hash。若只想检查参数展开，可加 `--dry-run`，此时不要求 `--model`/`--output`，
也不会写文件。

## 3. 饱和吞吐：dense 与 BTS 两臂

选择一张独占 GPU，然后运行两臂的正式配对协议：

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m eval.VLLM.run \
  --model "$MODEL" \
  --workload "$WORKLOAD" \
  --output /external/results/vllm-dense-vs-bts-saturated \
  --lengths 32768,65536,131072 \
  --request-rates inf \
  --restarts 5
```

`--request-rates inf` 是默认值，测量无限 offered load 下的
`request_throughput`。默认 `max_num_seqs=16`，`max_num_batched_tokens` 等于最长长度；
两臂均固定 BF16 model/KV cache、TP=PP=DP=1，并关闭 prefix caching 和 chunked
prefill。两臂还显式传入 `--attention-config '{"flash_attn_version":2}'` 与
`--enforce-eager`；runner 会从 worker 日志复核实际解析出的版本为 2。KV cache selector
使用 vLLM 0.17 原生的 `auto`；由于模型固定为 BF16，实际
KV storage dtype 仍为 BF16。显式传 `bfloat16` 会在该版本的 cache-update C++ op
warmup 中失败。两臂还统一设置
`PYTORCH_ALLOC_CONF=expandable_segments:True`（并同步设置 PyTorch 2.10 仍优先读取的
`PYTORCH_CUDA_ALLOC_CONF`），避免 128K 临时 FP32 分数矩阵因缓存分配器
碎片化而 OOM；该配置不改变 BTS 分数精度、Top-K、kernel 或请求调度。可用
`--pytorch-alloc-conf` 显式覆盖，但覆盖值不属于冻结正式协议。smoke 可用
`--num-prompts 1 --restarts 1`；少于 5 个 restart 或删减长度的
结果只能是 `INCONCLUSIVE`。正式运行不要使用 `--num-prompts` 截短冻结 workload。
`--dry-run` 只打印完整计划，不校验安装、模型、workload 或 GPU。

正式 `PASS` 还要求未使用 `--num-prompts` 或额外 CLI 参数、默认每 task 8 条、task 原生
输出预算、seed 42、一个 client warmup、`max_num_seqs=16`、
`max_num_batched_tokens=131072`、10,000 次 bootstrap 和本地冻结模型目录；任何偏离都会
被记录为 formal eligibility failure，并返回 `INCONCLUSIVE`，而不是用 smoke/pilot 冒充
正式结果。

runner 自己启动和停止 dense/BTS server，无需手工运行两个 `vllm serve`。dense 进程
禁用 BTS plugin 并选择 `FLASH_ATTN`；BTS 进程只启用 `bts_vllm` plugin 并选择
`CUSTOM`。不要用两个手工启动的 server 替代此入口，否则会丢失 fresh restart、交替
顺序、同请求校验和 backend 激活审计。

## 4. SLO goodput 容量

若实际服务声明关心延迟约束，应在实验前冻结一组共同的 offered request-rate 网格和
每个长度的 SLO。例如 `/external/config/vllm-slo.json`：

```json
{
  "32768": {"ttft": 5000, "tpot": 100, "e2el": 15000},
  "65536": {"ttft": 9000, "tpot": 100, "e2el": 22000},
  "131072": {"ttft": 18000, "tpot": 100, "e2el": 40000}
}
```

SLO 数值单位沿用 `vllm bench serve`：`ttft`、`tpot`、`e2el` 均为毫秒。上面的数字
只是格式示例，不是预先认可的门槛；正式值必须在看结果前按业务需求冻结。

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m eval.VLLM.run \
  --model "$MODEL" \
  --workload "$WORKLOAD" \
  --output /external/results/vllm-dense-vs-bts-goodput \
  --request-rates 0.05,0.1,0.2,0.4,0.8,1.6 \
  --slo-config /external/config/vllm-slo.json \
  --restarts 5
```

goodput 模式以满足 SLO 的 `request_goodput` 最大值作为容量。相同 rate grid 会用于
dense/BTS；最高 rate 必须明显进入过载区，否则容量未被网格包住，正式状态为
`INCONCLUSIVE`。先用独立 pilot 选择网格，再冻结网格做正式运行；不要根据正式结果为
不同方法分别调整 rates。

## 5. GPU 独占、统计判定与产物

正式硬件是 NVIDIA H100/SM90；BTS fixed-prefix fast path 不接受其他 compute
capability，结论也只应在记录的硬件范围内表述。`CUDA_VISIBLE_DEVICES` 必须精确指定
一张物理 GPU index 或 UUID。runner 不会杀死已有进程；若所选 GPU 在任一臂启动前
存在 compute process，会直接拒绝运行。每个 client 运行期间还会每秒采样 compute PID；
若发现不属于本次 server/client 进程树的 PID，会只停止自己的进程并使验收失败。输出
目录也必须是新目录或空目录。

统计单位是配对的 fresh-server restart。每个长度先计算每对 restart 的
`BTS capacity / dense capacity`，再执行 10,000 次单侧 95% paired bootstrap。只有
默认三个长度、至少 5 个完整配对 restart、rate grid 覆盖容量，且三个长度的 lower
bound 都 `>= 1.5`，才是 `PASS`。

根输出包括：

- `manifest.json`：运行合同、版本、模型/workload/source hash 和运行状态；
- `progress.json`：已完成的 restart pair 数；
- `benchmark_report.json`：三档配对比值、bootstrap lower bound 和总状态；
- `restart-XX/{dense,bts}/`：server/client log、实际命令、vLLM 详细结果和 BTS audit。

退出码为：

- `0`：正式 `PASS`；
- `2`：正式 `FAIL`，至少一个长度未证明 1.5x lower bound；
- `3`：`INCONCLUSIVE`，例如 restart 不足、长度不全或 rate grid 未覆盖容量；
- 其他非零值：安装、模型、workload、GPU、backend 审计或执行合同错误。

### 无调度器主机上的后台排队

若当前所有 H100 都被占用，可将一次性等待器放入 detached `tmux`，无需保持当前 shell
或人工持续查看。以下命令默认每 60 秒检查所有本机 H100 的 compute PID；同一张卡连续
3 次无 compute process 后，设置对应 `CUDA_VISIBLE_DEVICES` 并只启动一次正式 runner：

```bash
"$PYTHON_BIN" -m eval.VLLM.idle_queue \
  --queue-dir /external/results/vllm-dense-vs-bts-queue \
  --poll-seconds 60 \
  --stable-checks 3 \
  -- \
  --model "$MODEL" \
  --workload "$WORKLOAD" \
  --output /external/results/vllm-dense-vs-bts-saturated \
  --lengths 32768,65536,131072 \
  --request-rates inf \
  --restarts 5
```

`--gpu-indices 0,1` 可限制候选物理卡。`--queue-dir` 必须不存在，且与 runner 的
`--output` 不同。命令返回后可读取 `QUEUE/status.json` 和 `QUEUE/queue.log`；正式 runner
的标准输出写入 `QUEUE/runner.log`，详细 server/client 记录仍在 `--output`。运行中的
tmux session 名保存在 `QUEUE/launch.json`。

等待器使用完整 GPU UUID 同时取得 vLLM 自身锁，以及开发树/冻结提交两种 FastDeploy
队列锁命名，避免本项目的两套自动等待器在空闲窗口互相抢卡；随后按 vLLM 0.17 所要求的物理 GPU 编号设置
`CUDA_VISIBLE_DEVICES`；整个测试期间同时持有 GPU UUID 锁和 runner 输出路径锁。
本机锁只能防止由此 launcher 创建的队列互相抢卡，不能向其他用户预留 GPU。
因此选卡后 runner 仍会在每个 arm 前和 client 运行中检查全部 compute PID；若外部任务抢占，它只
停止自己的进程并将本次验收标为失败，不会结束外部进程，也不会在部分结果上自动重跑。

## 6. 准确率门槛与结论边界

serving runner 只验证每个请求完成且输入/输出 token 数与冻结 workload 一致；它不把
生成内容打分，因此吞吐 `PASS` 不能单独支持“保持准确率”。在声明 1.5x 前，还必须对
相同模型、BTS 配置、任务、样本和 seed 独立完成 dense/BTS 的 RULER（建议同时包含
LongBench）质量非劣化验收：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ruler.sh \
  --model "$MODEL" --attention dense \
  --output /external/results/ruler-dense-quality
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ruler.sh \
  --model "$MODEL" --attention bts \
  --output /external/results/ruler-bts-quality
```

稳定 RULER 入口一次只运行一个方法，不负责把这两份开发输出自动合并成 paired 声明。
质量门槛应在运行前冻结，例如逐长度对任务 score 做配对 bootstrap，并要求
`BTS - dense` 的单侧 95% lower bound 不低于预先声明的非劣化容忍值 `-δ`；`δ` 需由
项目方按任务风险定义，不能在看结果后选择。现有 RULER 比较器生成同任务的完整聚合和
逐样本证据，但不会替本项目自动决定 `δ`。

最后，这个两臂设计让 attention backend、模型和 stock vLLM scheduler 保持一致，
能回答“在 stock vLLM 中启用 BTS 后，完整系统吞吐是否达到 1.5x”。它不能回答“推理
调度模块本身是否贡献 1.5x”。若要做调度归因，必须增加并冻结四臂消融：

| Attention | Scheduler | 用途 |
|---|---|---|
| dense | stock | 基线 A |
| BTS sparse | stock | attention/system 效应 B/A |
| dense | new | scheduler 在 dense 下的效应 C/A |
| BTS sparse | new | scheduler 在 sparse 下的效应 D/B；完整方案为 D/A |

只有 `C/A` 或 `D/B` 才能归因于 scheduler；`D/A` 是完整系统相对 vLLM baseline 的
结果，不能把全部提升都写成调度模块贡献。
