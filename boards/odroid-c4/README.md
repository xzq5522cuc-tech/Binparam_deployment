# ODROID-C4 人体重建交接

插好 USB 摄像头和 HDMI。登录 odroid 图形桌面后自动启动，也可以双击桌面的「启动人体重建」。服务会等待 /dev/video0 出现。识别到人体后按 S 保存并打开三维查看器，查看器内按 Q 或 Esc 返回预览。预览内按 Q 或 Esc 退出。

| 路径 | 用途 |
| --- | --- |
| start-desktop.sh | 把当前显示会话传给用户级 pymaf-camera.service，再启动服务 |
| wait-for-camera.sh | 等待 USB 摄像头，随后执行 start-camera.sh |
| start-camera.sh | 配置线程与动态库，启动程序并防止重复占用摄像头 |
| tools/camera_auto_fast.py | 采集、推理调度、预览及结果保存 |
| tools/person_detector.py | YOLO 人体检测和跟踪 |
| tools/pose_actions.py、pose_actions_config.json | 姿态识别规则和阈值 |
| tools/view_saved_mesh.py、view_mesh_overlay.py | 三维网格查看和相机画面叠加 |
| bin/rpc_cap4 | C4 原版 ARM64 常驻推理后端 |
| lib/ | Bolt 引擎、张量、图像、矩阵、模型解析、公共库，以及 ONNX Runtime |
| models/bnn/feature.bolt、stage0–2.bolt | 图像特征及三个回归阶段 |
| models/person_detector/yolov8n.onnx | 人体检测模型 |
| models/smpl_ort/smpl_neutral_opset16.onnx | SMPL 网格重建 |
| source/data/ | SMPL 模板、网格采样映射及 PyMAF 平均参数 |
| outputs/ | 保存的图片、mesh.obj 和运行日志，启动错误见 camera.log |

旧工程与历史测试材料可从 Windows 的 ODROID-C4_pre_edit_20261113.tar.gz 恢复。板端全量 3DPW 数据仍在 /home/odroid/.binparam-development-archive/pymaf_final/dataset/3dpw_full。

旧开发目录集中归档在 ~/.binparam-development-archive。要使用依赖旧绝对路径的历史测试，先将归档中的 pymaf_final、bolt 和 eval_c4 恢复到用户目录原位置。

此 Git 目录保存 C4 前端源码和启动脚本。直接部署需要原板的 bin/、lib/、models/ 和 source/data/，以及用户级 pymaf-camera.service 和桌面入口。可从 Windows 的编辑前备份恢复原工程。测试入口见 [板端验证指南](../../docs/TESTING.md)。
