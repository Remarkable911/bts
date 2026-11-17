# Efficiency

稳定入口 `scripts/run_efficiency.sh` 复用父实验最佳完整 efficiency 结果的同一批
Qwen3-8B layer-12 attention replay：32K、64K、128K；batch 1、BF16；forced PyTorch
Flash SDPA、MInference、FlexPrefill、XAttention、BTS 共享完全相同的 module
arguments。每个方法跑
3 轮，每轮 3 次 warmup 和 8 次计时，最终口径是 per-round mean 的 median。

```bash
export EFFICIENCY_REPLAY_ROOT=/absolute/path/to/efficiency_replays
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN="$(command -v python)" \
bash scripts/run_efficiency.sh
```

默认测试 zero-based layer 12，需通过 `EFFICIENCY_REPLAY_ROOT` 或
`--attention-inputs-dir` 指向准备好的 frozen replay 目录。未指定数据来源时脚本会
给出提示并退出，不会查找开发者服务器上的目录。`--layer` 支持
空格和等号两种覆盖形式。显式测试其他层且未提供 replay 时，runner 会加载真实模型并
捕获指定层的输入，不会继续注入默认的 layer-12 replay。例如：

```bash
env -u EFFICIENCY_REPLAY_ROOT bash scripts/run_efficiency.sh \
  --layer 18 \
  --output "$PWD/results/efficiency/bts_topk040_tail256_layer18_capture.json"
```

如果已有其他层的 replay，也可以显式提供；runner 会校验 replay 中的实际层号，绝不会
把 layer-12 输入冒充成其他层：

```bash
bash scripts/run_efficiency.sh \
  --layer 18 \
  --attention-inputs-dir /absolute/path/to/layer18/replays \
  --output "$PWD/results/efficiency/bts_topk040_tail256_layer18_replay.json"
```

上述命令会直接执行完整计时并写入结果文件。replay 文件不随仓库发布，需要自行准备；
默认 layer-12 协议期望目录中包含 `replay_32768.pt`、`replay_65536.pt` 和
`replay_131072.pt`。如需从模型捕获自定义输入，可显式提供 `--prompt` 或 `--data-path`；
自定义输入不等同于已锁定哈希的历史 replay。

成功结束时，终端会从本次运行生成的 `summary_rows` 打印与父目录相同口径的固定宽度
汇总表，包含方法、长度、attention latency、相对 Dense 加速、峰值显存和 logical
density，并显示结果 JSON 路径。表格不会读取
`historical_reference` 来补数；失败异常仍直接抛出，不伪装成成功汇总。

runner 会逐个校验
`replay_{length}.pt` 的 batch/length、BF16、layer=12、模型 ID、模型 revision 与捕获输入
SHA256；不加载或执行父仓源码。

FlexPrefill 是稳定入口新增的当前测量臂，固定为 gamma `.95`、tau `.1`、block 128、
无 min/max budget、GQA interleave off。下述 2026-11-06 历史 four-way reference 不含
FlexPrefill，因此新运行的 FlexPrefill 数值只来自本次计时，不会从历史产物继承。
`reference_protocol_match.matches_defaults=false` 是新增该测量臂后的预期结果。

父目录参考产物是
`work/efficiency_current_fourway_strict_20261106/summary.json`，SHA256 为
`d7bf3f227a15d9844ac2644af684a2a3f98924220e2b7c0849c7c224ac71c423`。其中 BTS 的
32K/64K/128K latency median 为
`22.6069040 / 51.5108957 / 140.1575356 ms`，相对 forced Flash 为
`2.18718x / 3.35468x / 4.88126x`。

此历史产物未随仓库发布；如果拥有该产物，可用 `PRISM_REFERENCE_ROOT` 指定其所在
开发项目的根目录以启用来源哈希校验。缺少历史产物时，报告会如实记录
`source_present=false`。

这些历史值只写入 `historical_reference` 作为 provenance，
`reproduced_by_current_run=false`；新运行只报告自己的计时，绝不会把历史数值预填成新
结果。提交源码与历史开发源码并非 byte-identical，硬件或依赖变化也会改变计时，因此
配置一致不等于数值保证逐位相同。

测量范围是完整 attention module（QKV projection、Q/K norm、RoPE、attention、output
projection，以及 sparse selection/packing），不是端到端生成或 serving throughput。
`eval/efficiency/run.py` 保留作底层/开发诊断，但不是提交版稳定 shell 入口，也不能拿它的
synthetic layer-0 协议替代上述父目录结果。
