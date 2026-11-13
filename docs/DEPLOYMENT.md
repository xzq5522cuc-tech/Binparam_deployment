# BinParam 模型部署与验证

这份教程覆盖 **PyTorch → ONNX → Bolt → ODROID-C4 / Raspberry Pi 5**。训练服务器负责导出和生成参考结果，开发板负责推理和板端计时。

## 1. 从 PyTorch 导出 ONNX

使用 graduation/BinParam-main 的正式拆分导出入口生成 FeaturePyramid 和 Stage0–2。服务器路径、环境、checkpoint 校验值、旧 checkpoint 兼容处理及导出验证结果见 [PyTorch → ONNX 实操](PYTORCH_ONNX.md)。

## 2. 编译 X2bolt 并转换

本地历史版本使用 Bolt 基线 `b9a09d2` 加 [项目补丁](../patches/bolt_local_vs_b9a09d2.patch)。在 Linux ARM64 上准备 Git、C/C++ 工具链、CMake 和项目依赖。下列命令使用源码中的 `install.sh` 构建入口。构建依赖的下载与编译选项见脚本，完整板端构建需按各板归档核对。

```bash
git clone https://github.com/huawei-noah/bolt.git bolt
git -C bolt checkout b9a09d2
git -C bolt apply --check "$PWD/patches/bolt_local_vs_b9a09d2.patch"
git -C bolt apply "$PWD/patches/bolt_local_vs_b9a09d2.patch"
(cd bolt && bash install.sh --target=linux-aarch64_blank --converter=ON \
  --shared --openmp --test=OFF --threads=2)
```

X2bolt 的 `-d` 指向 ONNX 所在目录，`-m` 传不带扩展名的模型名，`-i` 选择转换精度。下面先转一份图，确认生成的 Bolt 文件非空。

```bash
X2BOLT="$PWD/bolt/install_linux-aarch64_blank/tools/X2bolt"
"$X2BOLT" -d /path/to/onnx -m feature_pyramid_opset16 -i FP32
```

批量转换建议用仓库的脚本。它要求 Python 3.11+，接受上面两套文件命名，保留转换日志和 SHA-256，不覆盖已有输出目录。

```bash
python3 tools/convert_onnx_to_bolt.py --converter "$X2BOLT" \
  --onnx-dir /path/to/fp32-onnx --mode FP32 --out artifacts/fp32
python3 tools/convert_onnx_to_bolt.py --converter "$X2BOLT" \
  --onnx-dir /path/to/fp32-onnx --mode FP16 --out artifacts/fp16
python3 tools/convert_onnx_to_bolt.py --converter "$X2BOLT" \
  --onnx-dir /path/to/bnn-export-onnx --mode BNN --out artifacts/bnn
(cd artifacts/bnn && sha256sum -c SHA256SUMS)
```

输出分别带 `_f32.bolt`、`_f16.bolt`、`_f16_b.bolt` 后缀。旧部署有些文件名简化为 `feature.bolt`、`stage0.bolt` 等，装配时按运行配置映射并记录哈希。BNN 需要对应的导出图、转换器和 ARM 算子修复。用一般 ONNX 指定 `-i BNN` 无法证明它与项目发布权重等价。SMPL 的 `smpl_neutral_opset16.onnx` 继续由 ONNX Runtime 执行。

## 3. 装配板端运行时

| 平台 | 已有入口 | 复现边界 |
| --- | --- | --- |
| ODROID-C4 | 历史布局 `/home/odroid/pymaf_final/final_release`，使用该板编译的 Bolt、ORT、`rpc_cap4` 和 trace 引擎 | 已验证现成运行时的样图推理和 OBJ 导出。从源码重编译仍需核对 C4 归档 |
| Pi 5 | 离线包解压到 `/home/cuc/BinParam-Camera`，包内含桌面安装脚本 | 完整离线包包含运行文件。重新编译需要原工程 `~/c4_migration/project`、Bolt 构建依赖及 ORT 1.23.2 ARM64 SDK |

Pi 5 历史工程在 `~/c4_migration` 下先用 `configure_pi.py` 从已有 Release CMake 配置生成 `project/pi_build/bolt`，再运行 `cmake --build project/pi_build/bolt -j2` 与 `python3 build_pi_runtime.py`。后者生成生产引擎、边界修复库和 `rpc_cap4`。这些脚本依赖完整历史工程和它的 CMake 缓存，本仓库仅保留交接入口和测试子集。部署后先以 `ldd` 检查共享库，再用固定输入检查能否返回有限的 6890×3 顶点。

## 4. 两板测试

详细命令、输入与结果见 [板端验证指南](TESTING.md)。先跑少量样本校验输出，再做精度统计。后端性能使用生产引擎，算子计时使用 trace 引擎，两组结果分别记录。跨板比较时固定模型、输入、预热和计时口径，同时报告各自的线程、绑核、运行库、温度与频率。
