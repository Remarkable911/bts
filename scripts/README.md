# Evaluation launchers

`scripts/` 只保留四个有父目录完整结果支撑的稳定入口。`PYTHON_BIN` 可指定评测环境；
调用方参数优先于脚本默认值。

| 入口 | 默认协议 | 已核验父目录参考 |
|---|---|---|
| `run_efficiency.sh` | forced Flash + MInference/FlexPrefill/XAttention/BTS；layer-12 frozen replay；32K/64K/128K；3×(3 warmup+8 runs)；threshold `.93` | BTS `22.6069/51.5109/140.1575 ms` |
| `run_longbench.sh` | 21 tasks / 4,750 samples；32K；Top-K `.4` + Tail256；threshold `.93` | task-macro `40.8784696310` |
| `run_ruler.sh` | 默认 frozen 5×3×100 = 1,500 samples；非 formal；只将 block threshold 改为 `.95` | 完整 13-task 历史参考 `82.2628205/75.1666667/69.6769231`，不作为 selected-5 汇总 |
| `run_density.sh` | exact official prompt/128K；base Qwen dense capture + 36 层 BTS replay；5 类各 10；threshold `.93` | `7.4343236%–11.1898202%`（3/5 类 `≤10%`） |

以下命令会启动完整 GPU 评测，并把新测结果写入各 runner 的默认输出目录：

```bash
PYTHON_BIN=/path/to/python bash scripts/run_efficiency.sh
PYTHON_BIN=/path/to/python bash scripts/run_longbench.sh
PYTHON_BIN=/path/to/python bash scripts/run_ruler.sh
PYTHON_BIN=/path/to/python bash scripts/run_density.sh
```

RULER 默认按顺序执行 `niah_single_1,niah_multikey_1,ruler_fwe,`
`ruler_qa_hotpot,niah_multiquery`。调用方可覆盖，例如：

```bash
PYTHON_BIN=/path/to/python bash scripts/run_ruler.sh \
  --tasks niah_single_2,ruler_qa_squad \
  --output "$PWD/results/ruler/custom_tasks"
```

默认不添加 `--formal`；完整正式评测需显式传
`--tasks ruler --formal --output /new/empty/directory`。

四个入口都会在成功结束时直接打印本次运行的结果汇总。LongBench/RULER 使用
lm-eval 的任务/分组表；Efficiency 打印方法 × 长度的实测 latency、相对 Dense
加速、显存和 density 表；Density 打印五个类别各自的样本数、平均 density、平均
pruning、标准差、pruning 范围及 `≤10%` 验收结果。Efficiency/Density 的终端表只读取
本次运行内存中的测量结果，历史 reference 不会被填入表格；完整审计字段仍保存在 JSON。

Efficiency 默认读取：

```text
/data3/lsf_data/prism_artifacts/efficiency_review_20261009_inputs
```

搬移后设置 `EFFICIENCY_REPLAY_ROOT`。RULER 和 Density 默认读取父目录的：

```text
results/ruler/topk040_tail256_official_protocol_20261105
```

搬移后设置 `RULER_SUITE_ROOT`；也可用 `PRISM_REFERENCE_ROOT` 改父目录基准。Density
还读取五个已锁 hash 的 official output JSONL，并固定 base
`Qwen/Qwen3-8B@b968826...`；suite YaRN4 只负责这些 prompt 的历史生成。脚本不执行父仓源码。

四项 reference 的 SHA256 依次为：

- Efficiency：`d7bf3f227a15d9844ac2644af684a2a3f98924220e2b7c0849c7c224ac71c423`
- LongBench：`9826abeecb30c7c6c1662361cf3bc0b0ec914f1dfa89bd0b2ed8042bc9522b81`
- RULER 完整 13-task 历史结果：`cc333bc31b199b2ccc1f01a6e586d15d649a71d906020f146107f432960bd406`
- Density：`9b2885529829062fa23d1bf747a09dd317a5eb6ee9baed1b61bfe3cd0e2663ca`

Density 固定五类依次为 NIAH single-1、NIAH multi-key-1、RULER frequent-word
extraction、RULER QA-Hotpot、NIAH multi-query；每类使用父实验相同的固定索引
`0,11,22,33,44,55,66,77,88,99`。前者和 frequent-word extraction 的历史 density
超过 10%，不会被改写成通过。父目录没有这两类的 full-100 Density 结果。

这些值不会被复制成新结果。Density 新跑会逐条比较全部 50 个 parent raw sample mean，
只有全部匹配才可标为 formal。RULER 的 `.95` 只改 block low/high
mass threshold；token Top-P 仍是 `.93`。LongBench、Density 和 Efficiency 保持 `.93`，
因此不存在一套诚实的“全任务共同最优 threshold”。

FastDeploy 的现有 formal 报告未在全部长度达到目标；vLLM 和 ThreeWay 没有完整 paired
formal 结果。因此它们没有稳定 shell 入口，相关 `eval/` 源码仅供开发验证。
