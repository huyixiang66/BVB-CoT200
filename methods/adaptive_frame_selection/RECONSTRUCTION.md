# V5 聚类后的位姿条件重构

方法链：问题物体解析 → 检测波峰及光流候选 → DINOv2 外观与 LingBot 位姿聚类 → 每组选择 4～8 帧 → 每组独立生成最简场景 JSON → 本地资产编译为 Blender → 在共享坐标系中合并 → 渲染双轴 20° 斜俯视图。

只上传方法、测试和依赖。视频、问题实例、图片、位姿、检测缓存、模型权重、API 响应、生成场景与网页均须自行提供或生成。

## 当前实现的含义

- 模型只输出目标资产、直接关联的白块和墙体；地板、屋顶及相机轨迹不由模型生成。资产内部结构、材质和渲染由本地代码实现。
- 每张输入帧对应同一个全局坐标系中的 LingBot 位姿。VLM 接收保留三位小数的位置、朝向与内参；完整精度矩阵另存，不被压缩覆盖。
- 可以给关键帧叠加原检测框及类别颜色；prompt 明确这些框只是候选提示，允许拒绝错误标签。
- JSON 验证或 Blender 执行失败才反馈修复；第一次成功编译后停止。该闭环验证可执行性，不是渲染图视觉优化或几何正确性认证。
- 各组从一开始就被要求在共享坐标系重构。因此合并脚本保留原世界坐标，组变换是单位矩阵，并附上完整精度参考相机。它**不是**给独立局部坐标片段重新估计 Sim3，也没有 ICP、墙体去重、门洞连通或布局优化。
- 位姿本身是预测结果，并已对齐到已有 BVB 相机轨迹；不是标定真值，尺度也不是可靠米制。检测、聚类或模型几何错误不会因合并自动修复。

## 安装

在仓库中的 `methods/adaptive_frame_selection` 下执行：

```bash
python -m pip install -r requirements-reconstruction.txt
docker build -t bvb-sandbox:latest -f docker/Dockerfile docker
```

主机另需 Blender（用于检查、预览和合并），Docker daemon 需可用。使用当前 Docker 配置；Colima 用户可自行设置 `DOCKER_HOST`。GPU 抽帧/特征依赖单独见 `requirements-gpu.txt`，下载权重可设置 `HF_ENDPOINT=https://hf-mirror.com`。

重构使用 OpenAI 官方端点，从环境变量 `LLM_API_KEY` 或本方法目录下未跟踪的 `.env` 读取密钥。可设置 `BVB_MODEL`（默认沿用 `gpt-6-astra`）、`BVB_MAX_OUTPUT_TOKENS`（默认 32768）、`BVB_COST_LIMIT_USD`（默认每组 5）。模型名须在使用的账号可用。

`high` 加 prompt 中 reasoning + final 共 2000 tokens 是软约束，不保证遵守。API 输出上限含推理与可见输出，也不保证在限额内返回有效 JSON。费用计算沿用历史实验费率，仅用于预算估计，不是实时价格或账单；每组最多 20 轮、约 1800 秒，实际单次请求可能跨过软预算。上传代码不触发付费调用。

## 输入约定

`--question-folder` 使用 `bvb_v5.scene_report` 输出的单题目录，包含公开 `question.json` 及 `instance-groups.json`（每组有 object、group_id、selected）。不读取标准答案。

`--cache` 包含 `candidates.json` 与原始帧，帧索引、时间戳、SHA256 必须和选帧记录一致。

`--poses` 包含：

- `camera-poses.npz`：校正后的 OpenCV C2W 与 intrinsics。
- `aligned-poses.npz`：`c2w_blender`，同一全局 Blender 坐标系。
- `alignment.json`：rotation、translation、scale_lingbot_to_blender、center_rmse_blender_units。
- `sampling.json`：video_sha256 及与候选缓存严格对齐的 frames（source_frame_index、timestamp_s、sha256）。
- `geometry-info.json`：image_size_wh，说明原内参对应图像尺寸。

准备过程会验证 `R_blender = R_alignment @ R_cv @ diag(1,-1,-1)` 和 `p_blender = scale * R_alignment @ p_cv + translation`。当前重构/参考相机协议沿用 **640×480** 输入，准备阶段会检查尺寸；其他分辨率需先适配协议，不要直接套用。

## 运行

先仅准备输入（不调用模型）：

```bash
python scripts/run_cluster_reconstruction.py \
  --question-folder /path/to/report/03-per-question/01-qID \
  --cache /path/to/detection/cache --poses /path/to/aligned/lingbot \
  --out /path/to/new/reconstruction --annotate-boxes --prepare-only
```

检查 `manifest.json`、各组 `selected-frames.jpg`、`pose-reference.json` 后，用相同参数移除 `--prepare-only` 执行。可指定 `--blender /path/to/blender`。已有成功结果会复用；部分付费调用记录会阻止自动重试，需先检查失败原因。`--finish-only` 仅重做已有场景预览和汇总。

各组生成 `assets/scene.blend`、`scene-spec.json`、`targets-highlight.png`、执行/验证记录及 API usage；汇总网页标明缺失目标。**缺失目标也可能得到可编译的墙体场景，不能只看 done 字段判定成功。**

将所有片段放入共享坐标系并渲染：

```bash
blender --background --disable-autoexec --python-exit-code 1 \
  --python scripts/merge_cluster_pose_scenes.py -- \
  --source /path/to/reconstruction --out /path/to/new/merged
```

输出 `merged-scene.blend`、`merged-top.png`、`merged-oblique.png`、带位姿中心的图及 `merge.json`。脚本检查来源位姿一致、网格世界包围盒未变、相机矩阵一致及原 blend 哈希未变。若不同片段原本采用不同局部坐标，这个入口会不适用。

## 测试

```bash
MPLCONFIGDIR=/tmp/bvb-mpl python -m unittest \
  bvb_v3.test_adaptive_peaks bvb_v5.test_selection \
  bvb_v5.test_dinov2_appearance bvb_v5.test_instance_sampling \
  bvb_v5.test_pose_association bvb_v5.test_v5_integration \
  bvb_v5.test_global_context
```

这些测试不需要 API、视频或模型。Docker 包装和镜像配方的来源及许可证见 `scripts/vendor/README.md`；V5 聚类上游提交见 `bvb_v5/UPSTREAM.json`。
