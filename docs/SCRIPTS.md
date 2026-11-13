# 脚本索引

路径相对于仓库根目录。这里列出模型导出、转换和验证所需的脚本。

| 路径 | 用途 |
| --- | --- |
| `export/pytorch/export_hybrid_onnx_bnnfc.py` | 加载训练配置和 checkpoint，导出 feature 和 stage0–2，检查 ONNX 并与 PyTorch 对比 |
| `export/pytorch/hybrid_partition_bnnfc.py` | 定义特征金字塔与回归子图，将二值全连接表达为 Bolt 可识别的卷积 |
| `export/pytorch/audit_hybrid_bnn.py` | 审计拆分图中的二值权重、Sign 边及 BNN 候选 |
| `tools/convert_onnx_to_bolt.py` | 批量调用 X2bolt，保存四份 Bolt 模型、转换日志及 SHA‑256 |
| `boards/c4/c4_eval.py` | 批量生成 C4 的 3DPW 顶点预测 |
| `boards/c4/benchmark_c4.py` | 测量 C4 常驻后端耗时、模块耗时和峰值内存 |
| `boards/c4/paired_ops_c4.py` | 用 C4 插桩引擎采集 FP32、FP16、BNN 算子日志 |
| `boards/c4/analyze_c4_ops.py` | 从两板兼容格式的日志中配对并汇总 54 个算子 |
| `boards/pi5/benchmark_pi5.py` | 复测 Pi 5 常驻后端耗时和重复推理输出哈希 |
| `boards/pi5/paired_ops_pi5.py` | 采集 Pi 5 不同精度与线程数下的逐算子日志 |
| `boards/shared/evaluate_3dpw_board.py` | 根据 3DPW 真值计算 PVE、MPJPE 和 PA-MPJPE |

`models_c4.example.json` 给出 C4 模型路径，`all54_targets.json` 定义算子匹配目标。`historical-*.json` 是历史统计。`patches/bolt_local_vs_b9a09d2.patch` 用于恢复项目的 Bolt 源码改动。
