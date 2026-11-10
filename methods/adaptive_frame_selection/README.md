# Question-conditioned adaptive frame selection

现已加入 **V5 外观＋位姿聚类 → 每组最简重构 → 共享坐标场景合并**。完整依赖、输入协议、命令和限制见 [RECONSTRUCTION.md](RECONSTRUCTION.md)，V5 选择规则见 [bvb_v5/README.md](bvb_v5/README.md)。下文保留 V3/V4 接口说明。

仅包含方法实现、测试、依赖与说明。没有实验结果、样例视频、图像、特征/位姿缓存、模型权重或生成网页。

## 方法组成

- `scripts/object_sampling/questions.py`：从问题模板解析相关物体，保留原解析函数的许可及来源元数据。
- `scripts/object_sampling/analyze_video.py`：GroundingDINO 检测、图像质量和光流分析。
- `bvb_v3/adaptive_peaks.py`：波峰发现、光流框关联、每峰候选抽帧。
- `bvb_v4/select_frames.py`：基于 LingBot 相机位姿的质量与视角互补选择。
- `bvb_v4/dinov2_appearance.py`：检测框裁剪、DINOv2 特征提取、跨帧外观相似度。
- `bvb_v4/instance_sampling.py`：多图匹配、保守/积极实例假设归并、每组帧预算。
- `bvb_v3/` 和 `bvb_v4/` 的报告脚本：生成可视化交付的代码，不包含生成产物。

## 归并规则

保守模式默认阈值0.82：两对一对一跨时刻匹配，双方锚点参与，每侧两张图间隔至少0.5秒；整组所有跨波峰对都须通过。几何冲突暂停合并。

积极模式默认阈值0.70，可用 `--threshold` 修改：不强制锚点参与，间隔至少0.25秒；组内存在一个共同代表直接匹配每个成员，全组波峰对均分至少0.60（缺失记0）；稀疏几何冲突仅警告。两种模式均禁止合并同帧明确独立的检测框，每组选择4～8帧。修改双图阈值不会同时修改组均分要求。

组是实例假设，不是物体真实数量；阈值不是身份概率。检测错误、遮挡和波峰内换目标都可能造成错误关联。

## 安装与测试

```bash
cd methods/adaptive_frame_selection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MPLCONFIGDIR=/tmp/bvb-mpl python -m unittest \
  bvb_v3.test_adaptive_peaks bvb_v4.test_selection \
  bvb_v4.test_dinov2_appearance bvb_v4.test_instance_sampling
```

测试不需要模型权重、API key 或实验数据。新跑模型特征提取需要 `requirements-gpu.txt`；模型可通过 `HF_ENDPOINT=https://hf-mirror.com` 下载。

## 调用

```bash
python -m bvb_v4.instance_sampling --help
python -m bvb_v4.dinov2_appearance --help
python -m bvb_v4.instance_report --help
```

使用自行准备的中间文件进行归并：

```bash
python -m bvb_v4.instance_sampling \
  --mode aggressive --threshold 0.4 \
  --v3 /path/to/v3/report \
  --v4 /path/to/v4/result \
  --appearance /path/to/dinov2/features \
  --cache /path/to/detection/cache \
  --lingbot /path/to/lingbot/poses \
  --out /path/to/new/output
```

输入约定：V3目录包含 `peaks.json`、`questions.json`；V4目录提供 `anchor-geometry.json` 和基线 `summary.json`；DINOv2目录包含 `rois.json`、`features.npz`、`extraction.json`；检测缓存包含 `candidates.json` 和 `config.json`；LingBot目录包含 `camera-poses.npz`、`sampling.json`。源帧索引、图像和特征哈希必须一致。输出目录须尚不存在。

当前 CLI 延续既有 V3/V4 中间文件结构；部分旧工具仍含场景路径、图像尺寸及采样数量假设。报告脚本使用 macOS 字体路径，迁移到其他平台或新数据需调整。该代码并非已验证适配任意视频的一键流水线。

第三方问题解析函数的许可证和来源记录位于 `scripts/object_sampling/vendor/`；V5 来源记录位于 `bvb_v5/UPSTREAM.json`。JSON 仅用于源码来源元数据，不包含实验结果。

## BVB V6 最新方法

[方法、输入格式和运行命令](bvb_v6/README.md)：物体框几何与 DINOv2 聚类、位姿互补选帧、无框 Astra 语义审核、每类候选覆盖保留、位姿条件最简重构与场景合并。仅方法代码和测试，无实验结果。

## V8 object-size reference mode

See [OBJECT_SIZE_REFERENCE.md](bvb_v6/OBJECT_SIZE_REFERENCE.md) for the single-cluster switch, visible reference and meter ruler, original-frame QA, and local uniform-FPS comparison report.

## V10 joint reconstruction

[V10 full workflow and commands](bvb_v10/README.md) packages the current per-question sampling/review front end and a single joint pose-conditioned scene generator. Use `python -m bvb_v10.workflow --help` for entities, raw-video features, run, QA, and reports. Historical V3–V8 modules remain as explicit dependencies and comparisons.
