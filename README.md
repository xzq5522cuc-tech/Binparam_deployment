# BinParam 板端部署交接

- [模型部署教程](docs/DEPLOYMENT.md)覆盖 PyTorch、ONNX、Bolt、ODROID-C4 和 Raspberry Pi 5。
- [板端验证指南](docs/TESTING.md)包含精度一致性、推理后端耗时和算子耗时。

ODROID-C4 的摄像头运行代码、桌面启动脚本和板端说明见 [C4 交接目录](boards/odroid-c4/README.md)。

## Pi 5 日常使用

插好 USB 摄像头和 HDMI 显示器。登录 Pi 5 的 cuc 图形桌面后，摄像头程序会自动启动。关闭后可以双击桌面的「BinParam 3D Camera」重新打开。如果桌面首次询问是否允许启动该图标，选择允许。

## 日常操作

1. 站到摄像头前，等待画面检测到人。程序在板上完成推理。
2. 按 S 或点击预览画面里的保存按钮，保存当前重建并打开三维网格查看器。
3. 查看器中可以拖动鼠标旋转，也可以用 A/D、W/S 改变角度、用 +/- 缩放、按 R 复位。按 Q 或 Esc 返回摄像头。
4. 在摄像头预览中按 Q、Esc 或关闭窗口退出。成果在本目录的 outputs/camera_*/snapshot_*/，mesh.obj 是三维网格，frame.jpg 是对应图像，prediction.npz 是推理数组，metadata.json 是保存时的参数与姿态标签。

## 文件清单

| 文件 | 用途 |
|---|---|
| README.md | 本手册 |
| start-camera.sh | 桌面和自启动共用入口，设置推理线程、库路径、USB 摄像头参数，防止重复启动 |
| tools/camera_auto_fast.py | 采集摄像头画面、调用后端重建、展示预览、保存并打开查看器 |
| tools/person_detector.py | YOLO 人体检测及短时跟踪 |
| tools/pose_actions.py | 从重建姿态识别动作 |
| tools/pose_actions_config.json | 动作识别阈值 |
| tools/view_saved_mesh.py | 保存后的交互式三维网格查看器 |
| tools/view_mesh_overlay.py | 可选查看器，将网格投影到保存时的相机画面 |
| bin/rpc_cap4 | 常驻 ARM64 推理后端，通过管道收取图像张量并返回姿态和顶点 |
| lib/libengine.so | 板端 Bolt 推理引擎 |
| lib/libtensor.so | Bolt 张量算子库，含 BNN 边界修复 |
| lib/libimage.so | Bolt 图像算子库 |
| lib/libblas_enhance.so | Bolt 矩阵计算库 |
| lib/libmodel_spec.so | Bolt 模型描述及加载库 |
| lib/libuni.so | Bolt 基础公共库 |
| lib/libonnxruntime.so.1.23.2 | ONNX Runtime，用于 SMPL 模型 |
| lib/libonnxruntime.so.1 | 上一文件的动态链接器别名 |
| lib/libonnxruntime_providers_shared.so | ONNX Runtime 执行提供器公共库 |
| models/person_detector/yolov8n.onnx | 人体检测模型 |
| models/bnn/feature.bolt | PyMAF 图像特征提取模型 |
| models/bnn/stage0.bolt | 初始姿态、体型和相机参数回归 |
| models/bnn/stage1.bolt | 第一轮参数细化 |
| models/bnn/stage2.bolt | 第二轮参数细化 |
| models/smpl_ort/smpl_neutral_opset16.onnx | 从姿态及体型参数生成人体网格的 SMPL 模型 |
| source/data/smpl_neutral_f32.bin | SMPL 人体模板与网格拓扑，供推理和 OBJ 导出使用 |
| source/data/smpl_dmap_431x6890.bin | 从 6890 个网格顶点采样到 431 个点的映射 |
| source/data/pymaf_mean_params_f32.bin | PyMAF 初始平均参数 |
| outputs/ | 自动生成的照片、网格、推理结果和日志。camera.log 记录启动错误，隐藏的 .camera.lock 用于防止重复启动 |

## 启动配置和故障排查

- 桌面快捷方式在桌面目录的 binparam-camera.desktop。用户级自启动配置在 ~/.config/autostart/binparam-camera.desktop。两者都调用 start-camera.sh。
- 开机停在登录界面时，先登录 cuc。程序依赖图形桌面，登录后启动。
- 如果打不开，先确认摄像头已插好、没有其他程序占用。默认使用 /dev/video0。启动错误见 outputs/camera.log。
- 从图形桌面的终端也可以执行 /home/cuc/BinParam-Camera/start-camera.sh。现有 Pi 5 系统提供 Python 3、OpenCV、NumPy 和基础系统共享库。
- 开发源码、实验和旧结果存放在 /home/cuc/c4_migration，日常使用只需要本目录。

## 恢复安装

Git 仓库保存脚本、配置、手册及运行文件校验表。完整离线交接包另外包含 ARM64 后端、动态库和模型。将离线包解压到 Pi 5 的用户目录后，进入 BinParam-Camera，执行 bash install-desktop.sh 安装桌面图标和登录自启动。可用 sha256sum -c RUNTIME_SHA256SUMS.txt 检查运行文件。

- install-desktop.sh 安装桌面图标和登录自启动。
- RUNTIME_SHA256SUMS.txt 列出离线运行文件及校验值。
