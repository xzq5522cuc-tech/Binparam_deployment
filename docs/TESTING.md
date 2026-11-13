# 板端验证指南

| 要验证的内容 | 保留脚本 | 结果怎么读 |
| --- | --- | --- |
| 与 3DPW 真值的精度 | [evaluate_3dpw_board.py](../boards/shared/evaluate_3dpw_board.py) | PVE、MPJPE、PA-MPJPE（mm），保留每样本结果 |
| 常驻推理后端耗时 | [C4 benchmark_c4.py](../boards/c4/benchmark_c4.py)、[Pi 5 benchmark_pi5.py](../boards/pi5/benchmark_pi5.py) | 三轮、每轮 5 次预热和 30 次计时，汇报均值、P95 和峰值 RSS |
| 逐算子耗时 | [C4 paired_ops_c4.py](../boards/c4/paired_ops_c4.py)、[Pi 5 paired_ops_pi5.py](../boards/pi5/paired_ops_pi5.py) | 用插桩库按算子配对，分别统计 feature 48 层、回归阶段 6 层及合计 54 层 |

## 3DPW 精度

各后端使用同一个 manifest 和预处理后的输入。每个输入是 602,112 字节的 little-endian float32（1×3×224×224）。每条预测按样本 ID 保存为 `<id>.pred_vertices.bin`，应为 82,680 字节且顶点有限。评分包另需真值顶点、`J_regressor_h36m_f32.bin` 和 `H36M_to_J14_i32.bin`。

C4 命令依赖历史工程中的可配置线程版后端 tools/analysis/openmp_candidate/rpc_threads。先用少量样本检查输出，正式评分时使用与实际预测完全对应的 manifest。Pi 5 的全量流式采集依赖外部数据与板端工具，当前收录两板共用的真值评分入口。

```bash
mkdir -p results
python3 boards/c4/c4_eval.py --pack /path/to/pack \
  --out results/c4-smoke --mode bnn --threads 2 --affinity 0-1 --limit 8
python3 boards/shared/evaluate_3dpw_board.py --pack /path/to/matching-pack \
  --predictions results/c4-smoke --output results/metrics.json
```

比较两板的 3DPW 指标时，使用相同的样本清单、真值及评分配置，并记录输入和模型哈希。

## 后端耗时

关掉摄像头和其他推理任务。用生产库启动常驻后端，同一模型组交替顺序跑三轮。计时包含特征提取、三个回归阶段和最终 SMPL，排除初始化、摄像头采集、预处理、显示及文件 I/O。

- **ODROID-C4**。脚本沿用原工程绝对路径，参考 [模型清单](../boards/c4/models_c4.example.json) 检查模型位置。历史默认 4 线程，曾出现卡住。当前脚本使用可配置线程的 rpc_threads，先用下面的 2 线程命令。与历史 4 线程数据比较时注明差异。
- **Pi 5**。脚本依赖 `~/c4_migration` 完整历史目录中的 FP32、FP16、BNN 权重、`rpc_cap4`、生产库和 3DPW 固定样本。输出默认在该目录下的 `precision_retry_20261109/`，复测请给新的输出路径。

```bash
python3 boards/c4/benchmark_c4.py \
  --models boards/c4/models_c4.example.json --out results/c4-backend \
  --threads 2 --affinity 0-1
BINPARAM_BENCHMARK_OUT="$PWD/results/pi5-backend" python3 boards/pi5/benchmark_pi5.py
```

## 逐算子统计

C4 使用原工程的 `trace_engine/libengine.so`，放在生产引擎和普通 Bolt 库之前。采集脚本会检查实际加载库。下例采集三种模型的 1 线程对照，分析器用 [54 层配对定义](../boards/shared/all54_targets.json) 处理三轮日志。

```bash
python3 boards/c4/paired_ops_c4.py \
  --models boards/c4/models_c4.example.json --out results/c4-paired \
  --threads 1 --affinity 0-1
python3 boards/c4/analyze_c4_ops.py --logs results/c4-paired --threads 1 \
  --targets-from boards/shared/all54_targets.json --out results/c4-all54.json
```

Pi 5 采集依赖原工程的 precision_profile/lib 插桩库。下面复用同一分析器处理 Pi 5 的配对日志，已用历史日志验证 54 层统计结果。

```bash
BINPARAM_BENCHMARK_OUT="$PWD/results/pi5-paired" python3 boards/pi5/paired_ops_pi5.py
python3 boards/c4/analyze_c4_ops.py --logs results/pi5-paired --threads 1,4 \
  --targets-from boards/shared/all54_targets.json --out results/pi5-all54.json
```

每轮从 40 条插桩记录里取最后 30 条。统计的是算子调用耗时，包含该算子内的布局处理或边界修复开销。正式归档保留模型与输入哈希、线程/绑核、频率/温度、`protocol.json`、`runs.json`、`summary.json` 及原始日志。仓库只留少量历史汇总，原始日志与模型从各自板端备份恢复。
