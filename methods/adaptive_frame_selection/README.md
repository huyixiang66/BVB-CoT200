# Question-conditioned adaptive frame selection

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

第三方问题解析函数的许可证和来源记录位于 `scripts/object_sampling/vendor/`；其中唯一的 JSON 是源码来源元数据，不是实验结果。

## 扩框和位姿关联

新增参数使用以下流程。准备外观缓存时保持原始检测框坐标，只有特征裁剪扩框。
小框面积不超过画面 2% 时每边扩 75%，其他框扩 20%，并在图像边界截断。

```bash
python -m bvb_v4.dinov2_appearance prepare \
  --crop-policy adaptive-context --out /path/to/context-appearance \
  --result /path/to/v4/result --peaks /path/to/v3/report/peaks.json \
  --cache /path/to/detection/cache
python -m bvb_v4.dinov2_appearance extract \
  --out /path/to/context-appearance --model facebook/dinov2-base --device cuda:0
python -m bvb_v4.instance_sampling \
  --mode aggressive --threshold 0.4 --pose-association --intrinsics-size 518 392 \
  --v3 /path/to/v3/report --v4 /path/to/v4/result \
  --appearance /path/to/context-appearance --cache /path/to/detection/cache \
  --lingbot /path/to/lingbot/poses --out /path/to/new/output
```

位姿阶段另外需要检测缓存的 `detections.jsonl`、原帧和 LingBot 的
`sparse-geometry-corrected.npz`。后者包含世界坐标 `points` 和点来源的采样索引
`frame_ids`。位姿必须是 OpenCV camera-to-world，与点云处于相同坐标系。
`--intrinsics-size` 是内参对应图像的宽高，投影时会按原帧尺寸缩放。

程序先完成原外观分组，再依次检查跨组证据。两份点云各有至少 24 个点且至少 60%
同时投进另一段相隔至少半秒的两帧原检测框，并且锚帧相似度达到 0.35，可以提供深度关联。
深度不足时，局部特征需要通过双向匹配、位姿三角化、相机前方检查和重投影检验，
并有跨三个视角的重复点，最大误差不超过 12 像素。每次新增合并都要求一条直接跨当前两组的证据。

两个波峰的两对匹配若四个裁剪都是扩 75% 的小框，可以使用 0.45 组门槛。
原外观分组仍使用 0.60 组均分和同帧分离框约束。新增几何合并的 0.60 排序优先级以
`priority_only` 标记，实际观测分数与几何记录分别保存在 `peak-association.json` 中。

新增回归检查可运行 `python -m unittest bvb_v4.test_pose_association`。
三题冻结缓存迁移检查保留相同组别和帧号。阈值是在单场景开发题上检查的经验参数。
计数视角的运行方式与结果范围见仓库根目录 `docs/pose-clustering-counting.md`。
