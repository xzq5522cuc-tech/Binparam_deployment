# PyTorch → ONNX

在校园网或 EasyConnect 内用 VS Code Remote SSH 打开 pymaf-server 上的 /data/users/xzq/graduation/BinParam-main。本节命令在该项目根目录执行。服务器源码基线为 1fca4fe，工作区有本地改动，单独的提交号无法重建当时的工作区。

## 导出入口（含原训练工程的历史脚本）

| 文件 | 用途 |
| --- | --- |
| export_deployment_onnx.sh | 统一入口，split 导出四个子图，full 导出完整图 |
| export_hybrid_onnx_bnnfc.py | 拆分导出、ONNX 检查和 PyTorch / ONNX Runtime 数值比较 |
| hybrid_partition_bnnfc.py | FeaturePyramid 和三个回归包装模块，以及 Bolt 所需的二值卷积改写 |
| deployment/audit_hybrid_bnn.py | 检查拆分图的二值权重和 Bolt BNN 候选算子 |
| export_bolt_candidates.py、make_folded_truebnn.py、validate_bolt_onnx.py | 完整图导出、二值权重改写、数值验证 |
| export_smpl_onnx.py | 单独导出由板端 ONNX Runtime 运行的 SMPL 图 |
| deployment/precision_20261020/export_biasfix_stages.py | 历史发布版三阶段偏置修复，需要 precision_reference.py 及 oracle 输入 |

## 复现拆分 BNN 导出

在训练项目根目录运行。先激活实测环境 `pymafx`（Python 3.8、PyTorch 2.1.0+cu121、ONNX 1.17.0、ONNX Runtime 1.19.2）。导出固定 batch 1、NCHW 1×3×224×224、opset 16。checkpoint 为 `checkpoints/model_weights.pth`，SHA-256 为 `e4e5f704163c6408731c2f171d3b245206090dbd4aafec6e8fb6c3bedd2f7cd3`。

将交接仓库放到训练服务器，替换下面的路径。导出脚本已兼容旧 checkpoint 的三个 `entropy_reg.bin_centers` 缓冲区，并检查其余权重键。

```bash
set -e
handoff=/path/to/Binparam_deployment_handoff
out=$(mktemp -d /tmp/binparam-onnx-XXXXXX)
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
  python "$handoff/export/pytorch/export_hybrid_onnx_bnnfc.py" \
  --variant bnn --checkpoint checkpoints/model_weights.pth \
  --cfg_file configs/binparam_pymaf_res50_mix_best.yaml \
  --output_dir "$out" --opset 16 --bolt_bnn_linear_as_conv
python "$handoff/export/pytorch/audit_hybrid_bnn.py" "$out"/*.onnx > "$out/audit_bnn.log"
(cd "$out" && sha256sum ./*.onnx > SHA256SUMS && sha256sum -c SHA256SUMS)
```

四份输出分别为 feature_pyramid_opset16.onnx 和 stage0/1/2_opset16.onnx。后者的输入是采样点特征和上一步的姿态、体型、相机参数，输出更新后的三组参数。板端用 C++ 执行阶段之间的 SMPL、投影和特征采样。审计日志可定位 48 个主干和 6 个回归阶段的二值算子候选，实际 ARM 内核派发需在板上检查。

本次在服务器用临时进程应用上述兼容检查并成功导出四份图，四份文件通过 SHA‑256 检查，导出器内置的 PyTorch / ORT 数值检查也通过。本次产物为基础拆分图。板端 c4_20261020_biasfix 的三份 stage 来自后续偏置修复流程，复现该发布版本时要单独核对修复后的三份 ONNX 和 Bolt 权重。本次实测范围为基础拆分 BNN，完整图、FP32 和偏置修复尚未重跑。
