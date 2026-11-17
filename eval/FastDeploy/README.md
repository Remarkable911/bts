# FastDeploy dense 与 BTS 吞吐实验（dev-only）

> **开发态说明：** 本目录没有可作为提交“最优结果”的完整成功证据，且对应的
> `scripts/` 启动器已移除、当前没有稳定脚本入口。以下内容仅保留实验设计与开发记录，
> 不应被引用为提交结果或可直接复现的正式验收流程。

本评测只比较同一 FastDeploy 2.5 引擎中的两条完整 prefill 路径：

- `dense`：FastDeploy 原生 dense FlashAttention kernel；仅在 PaddleFormers
  已经应用 RoPE 时增加 identity-RoPE 兼容 guard；
- `BTS`：本提交的稀疏 prefill adapter，并使用相同的语义兼容 guard。

该 guard 修正 FastDeploy 2.5 PaddleFormers FlashAttention 路径重复应用 RoPE
的问题，不替换或优化 dense attention kernel。两臂使用完全相同的模型、确定性合成
token IDs、引擎参数和 1-token generation。worker 内 backend 注册不可逆，因此 dense 与
BTS 每一臂都必须在 fresh process 中运行。

## 正式验收协议

正式验收按下列冻结合同执行：

- 模型必须是 FastDeploy/PaddleFormers 可加载、未量化的精确 Qwen3-8B BF16
  YaRN4 导出：36 层，hidden size 4096，32 个 query heads，8 个 KV heads，head
  dimension 128，`max_position_embeddings=131072`，`rope_type=yarn`，factor 4，
  original length 32768。仅有 Hugging Face YaRN overlay 不能替代该导出。
- 硬件与工作负载限定为一张独占 NVIDIA H100（compute capability 9.0）、单活跃请求、
  TP=DP=1、引擎容量 `max_num_seqs=16` 的离线顺序生成。关闭 prefix caching、chunked
  prefill 和 CUDA Graph；每个请求执行完整 prefill，只生成 1 个 token，因此本协议是
  prefill 主导测量，但不会执行第二个 token 所需的真实 decode iteration。需要审计原生
  decode 的统一三比值验收使用 `eval/ThreeWay` 的 2-token 配置。该 runner 不把顺序请求
  描述成并发 serving 吞吐。
- dense prefill 明确固定为 FlashAttention 2。适配器在 launcher 和 GPU worker 中同时
  固定 FastDeploy 的 `FLASH_ATTN_VERSION=2` 与 Paddle 的
  `FLAGS_flash_attn_version=2`；只设置后者不足以覆盖 H100 上 FastDeploy 默认的 FA3
  选择。decode 保持两臂共同的原生 `append_attention` 路径。
- 32K、64K、128K 指总上下文预算 32768、65536、131072。FastDeploy 2.5 要求
  `input_tokens + min_tokens < max_model_len`；为 1 个输出位置和 1 个引擎边界位置
  各预留一位后，实际输入长度固定为 **32766、65534、131070 tokens**。
- 正式序列固定为 5 个配对 fresh-process restart，并逐次交替 dense/BTS 运行顺序。
  每个 restart、每一臂、每个长度先执行 1 次 warmup，再执行 5 次 measured request。
  这 5 次请求只用于稳定该 restart 的中位数，不是 5 个独立统计样本；统计单位是完整
  restart pair。确定性合成输入与生成均固定 `seed=20261110`。
- 控制器在开始时生成运行时源码清单及 SHA-256，并将 canonical BTS config 的
  SHA-256、配置文件 SHA-256、runner SHA-256 和模型 `config.json` SHA-256 写入
  `manifest.json`。影响两臂的运行时源码在每一臂前后重新核对，源码若在运行期间变化
  则整次运行作废；所有 dense/BTS 臂和 restart 的 runtime-source manifest hash 与
  canonical BTS config hash 必须完全一致。
- “dense 为未修改的 FastDeploy 原生路径”不只依赖包版本：仓库内还冻结了实际批准的
  FastDeploy 2.5.0、PaddleFormers 1.1.1 与 PaddlePaddle 3.3.1 核心源码 SHA-256
  清单，覆盖 dense FlashAttention backend、engine args、PaddleFormers bridge、
  attention selector、Qwen3/YaRN 路由及 Paddle FlashAttention wrapper。controller、
  launcher worker 和真实 GPU executor 会各自重新定位并哈希实际加载文件，核对 AST
  语义合同与运行时模块路径；三方 digest 必须一致且命中批准清单，否则 fail closed。
- FastDeploy 构造引擎时会执行启动 profile。GPU worker 使用私有 `/dev/shm` mmap
  维护定长审计计数器；runner 在引擎构造和启动 profile 完成后读取基线，在全部
  warmup/measured 请求完成后再读取终值。每层请求审计数一律使用“终值减启动基线”的
  delta，因此启动 profile 产生的计数被明确扣除，不能冒充 benchmark 请求；delta
  必须与 3 个长度 ×（1 warmup + 5 measured）逐层精确对应。
- 启动每一臂前，所选物理 GPU 必须没有其他 compute process。臂运行期间控制器每
  1 秒通过只读 `nvidia-smi` 抽样检查 GPU compute PID，并在 launcher 退出后执行一次
  最终检查；以 benchmark 主进程及其
  后代进程作为允许集合；发现集合外的 PID 后会在 0.25 秒后重新解析后代并复查，以
  排除 worker 刚启动时的进程表竞态；若外部 PID 仍存在，就立即终止该臂并使验收失败。
  每一臂至少要有一次运行期检查，检查记录和观察到的 benchmark PID 持久化到
  manifest。
- 每臂还会注入唯一、不可复用的进程环境 token。即使 FastDeploy GPU worker 再次
  `setsid` 或 launcher 因 OOM/信号异常退出，controller 也只会根据该 token 清理本臂
  遗留进程，不会触碰 GPU 上原先存在的其他用户进程。

模型加载、引擎初始化和启动 profile 均不在计时区间内。对 restart `r` 和长度 `L`，
配对加速比定义为：

```text
median dense latency[r,L] / median BTS latency[r,L]
```

两臂输入长度相同时，这也等于 BTS input-token throughput 除以 dense input-token
throughput。长度级点估计是 5 个配对 restart ratio 的中位数，但点估计达到 1.5
并不足以通过。正式判定固定执行 **10,000 次单侧 95% paired-cluster percentile
bootstrap**：每次重采样完整 restart pair，并对三个长度共享同一组 restart 索引。
bootstrap seed 固定为 20261110。32K、64K、128K 的 bootstrap lower bound 必须
分别都满足 **>= 1.5**，整次验收才是 `PASS`；任一长度未达到即为 `FAIL`。

## 运行

先把本提交安装到启动 FastDeploy 的同一个解释器，再在独占 H100 上运行：

```bash
/path/to/fastdeploy-2.5-venv/bin/python -m pip install \
  --no-deps --no-build-isolation -e /path/to/prism-token-sparse-submission

CUDA_VISIBLE_DEVICES=0 \
/path/to/fastdeploy-2.5-venv/bin/python -m eval.FastDeploy.run \
  --model /path/to/Qwen3-8B-YaRN4-paddleformers \
  --output /path/to/new-results/fastdeploy-dense-vs-bts
```

仅设置 `PYTHONPATH` 虽然可以导入模块，却不会注册每个新 FastDeploy GPU worker
需要发现的 `bts -> bts_fastdeploy.plugin:register` distribution entry point。runner
会在等待 GPU 或加载权重前校验 `PYTHON_BIN` 中的该入口；若缺失则直接失败并给出安装
命令。

`CUDA_VISIBLE_DEVICES` 必须精确解析为一张物理 GPU。runner 不会终止运行前已存在的
进程；若 GPU 不独占则拒绝启动。H100 正式协议固定使用 Paddle/FastDeploy
FlashAttention 2；
`--paddle-flash-attn-version 2` 只供 backend 选择 FA2 的 pre-Hopper 非正式运行。
输出目录必须是新目录或空目录。

`--gpu-idle-timeout-seconds` 也覆盖 NVIDIA driver/device 暂时不可用的情况。在限定
时间内 runner 只读重试 `nvidia-smi`，只在所选 GPU 没有 compute process 后启动；
超时则 fail closed。

提交版不再提供自动串联 smoke、pilot 和 formal 的 shell suite。开发者如需继续实验，
应直接调用上述 `eval.FastDeploy.run`，显式选择长度与 restart 数，并分别保存输出；不得把
这些开发运行当作稳定提交结果。例如单次完整配置的 dry-run：

```bash
CUDA_VISIBLE_DEVICES=0 \
/path/to/fastdeploy-2.5-venv/bin/python -m eval.FastDeploy.run \
  --model /path/to/Qwen3-8B-YaRN4-paddleformers \
  --output /path/to/new-results/fastdeploy-validation \
  --lengths 32768,65536,131072 --restarts 5 --dry-run
```

控制器为每一臂写入 log、worker report 和命令记录，并生成 `manifest.json` 与
`benchmark_report.json`。GPU worker 在模型加载前写入不可变 provenance metadata，
运行时只更新 mmap 定长计数器；FastDeploy 结束 worker 后，控制器把合并后的 JSON
审计证据保存在 worker report 旁边。该证据不依赖 Python `atexit`、shutdown signal
或跨进程 trigger file，计时区间内也没有逐层文件写入。

退出码为：

- `0`：正式 `PASS`；
- `2`：正式 `FAIL`，即至少一个长度的 lower bound 未证明 1.5x；
- `3`：`INCONCLUSIVE`，包括少于 5 个完整 restart pair 的运行；
- 其他非零值：执行错误或协议合同错误。

## 结论边界

本协议只支持“FastDeploy 2.5 内、H100 单卡、单请求、prefill 主导的离线 dense/BTS
吞吐对比”这一结论。它**不是**并发 serving 吞吐、FastDeploy 调度器吞吐、vLLM
对比或实际任务 workload 的结论，也**不测试准确率或质量**。要声称“吞吐至少 1.5x
且保持准确率”，还必须在相同模型和 BTS 配置下另行冻结并完成 paired
RULER/LongBench 质量非劣化验收。logical density 或 token-retention 设置也不能解释
为本协议实测的 FLOP 裁剪比例。
