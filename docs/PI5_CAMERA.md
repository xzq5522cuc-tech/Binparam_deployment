# Raspberry Pi 5 摄像头演示

在 Pi 5 图形桌面双击 **BinParam 3D Camera**。识别到人后按 **S** 保存网格，按 **Q** 或 **Esc** 退出当前窗口。

恢复部署时，使用完整的 Pi 5 离线包。在解压后的 `BinParam-Camera` 根目录执行 `sha256sum -c RUNTIME_SHA256SUMS.txt`，再用包内的 `bash install-desktop.sh` 安装桌面快捷方式和登录自启动。启动日志见 `outputs/camera.log`。

本仓库的相机源码采用 `tools/camera.py` 和 `tools/detector.py`。离线包保留打包时的脚本名和配套启动入口，恢复部署时应成套使用。
