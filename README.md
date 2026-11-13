# BinParam 板端部署

从 PyTorch 权重导出 ONNX，经 Bolt 转换后部署到 ODROID-C4 和 Raspberry Pi 5。

| 步骤 | 输入 → 输出 | 从哪里开始 |
| --- | --- | --- |
| 1. PyTorch → ONNX | checkpoint + 模型配置 → FeaturePyramid 和 Stage0–2 四份 ONNX | [导出命令、依赖与校验](docs/PYTORCH_ONNX.md) |
| 2. ONNX → Bolt | 四份 ONNX → FP32、FP16 或 BNN Bolt 模型 | [编译 X2bolt、转换及校验](docs/DEPLOYMENT.md#2-编译-x2bolt-并转换) |
| 3. 装配板端运行时 | Bolt 模型 + SMPL ONNX + 本板 ARM64 后端和动态库 → 可运行目录 | [板端装配](docs/DEPLOYMENT.md#3-装配板端运行时) |
| 4. ODROID-C4 验证 | 固定输入 → 顶点预测和计时日志 | [C4 测试](docs/TESTING.md) |
| 5. Raspberry Pi 5 验证 | 相同模型与测试输入 → Pi 5 后端和算子计时 | [Pi 5 测试](docs/TESTING.md) |
| 6. 对比两板 | 预测顶点、真值和计时日志 → 3DPW 精度、后端及算子耗时 | [测试命令与统计口径](docs/TESTING.md) |

[脚本索引](docs/SCRIPTS.md)逐个说明仓库中的脚本。两板的摄像头演示通过板载桌面入口启动。Pi 5 离线包的恢复与校验见 [演示说明](docs/PI5_CAMERA.md)。

仓库包含拆分导出脚本、模型转换工具和测试入口。PyTorch 导出依赖原训练项目和 checkpoint，板端安装依赖各自的模型与运行库备份。基础 BNN 拆分图已在服务器上通过 PyTorch 与 ONNX Runtime 数值校验。板端发布的 `biasfix` 模型还有三阶段偏置修复步骤，完整复跑范围见导出教程。
