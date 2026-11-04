# BVB V5：外观与位姿联合聚类

本地版本 `5.0.0`，来自 [BVB-CoT200 的 pose-clustering-counting 分支](https://github.com/huyixiang66/BVB-CoT200/tree/pose-clustering-counting)，固定提交 `8c97b83a2d2492418442d97c4382eb0afc220d81`。源码对应关系见 [UPSTREAM.json](UPSTREAM.json)。V4 保留原样。这里是方法代码，不包含实验结果。

## 与 V4 的变化

1. **外观裁剪增加上下文**：面积不超过画面 2% 的检测框每侧扩展框宽/高的 75%，其他框每侧扩展 20%，越界截断。只改变 DINOv2 输入裁剪，原检测框不变。
2. **LingBot 三维关联**：把框内稀疏点投影到其他观测，检查共同覆盖；同时用 SIFT 匹配与三视角三角化验证跨帧关联。符合条件的光流恢复观测也可参与几何验证。
3. **新增合并需要直接证据**：每次跨组归并必须有新增跨组位姿关联，或满足扩框小物体配对条件，防止借已有组内关联无限扩张。同帧明确独立框仍禁止合并。
4. **计数渲染工具**：提供全局图与最多四张局部图，同一实例各部件保持同色。

V5 默认 `aggressive`、外观双图匹配阈值 **0.40**、启用位姿关联及上下文裁剪。0.40 延续此前十题实验的设置，是本地默认值调整；上游原 CLI 默认值不同。普通外观组仍要求代表关联和组均分至少 0.60；新增位姿证据可以绕过外观均分要求。位姿证明中的 `priority_only: true` 表示 0.60 只是排序常数，报告不会将它展示为实测相似度。每个实例假设保留 4～8 帧，必要时包含明确标注的未验证上下文帧；假设组数不等于真实物体数量。

## 调用

以下命令从本项目根目录执行。继续复用现有 `bvb_v3` 的候选生成与问题解析代码。CPU 依赖见 `requirements.txt`，DINOv2 推理额外依赖见 `requirements-gpu.txt`。

先为扩框策略创建**新**外观缓存，再提取特征；旧 tight-crop 特征不能直接视为扩框特征：

```bash
python -m bvb_v5.dinov2_appearance prepare \
  --result /path/to/v4/result --peaks /path/to/v3/report/peaks.json \
  --cache /path/to/detection/cache --out /path/to/new/appearance

HF_ENDPOINT=https://hf-mirror.com python -m bvb_v5.dinov2_appearance extract \
  --out /path/to/new/appearance --device cuda:0 --batch-size 32 \
  --revision f9e44c814b77203eaa57a6bdbbd535f21ede1415

python -m bvb_v5.instance_sampling \
  --v3 /path/to/v3/report --v4 /path/to/v4/result \
  --appearance /path/to/new/appearance --cache /path/to/detection/cache \
  --lingbot /path/to/lingbot/poses --out /path/to/new/v5/result
```

`--v4` 只是兼容的输入参数名，用于读取 `anchor-geometry.json` 与基线 `summary.json`。LingBot 目录需包含 `camera-poses.npz`、`sampling.json` 和 `sparse-geometry-corrected.npz`，相机数据须为已校正的 C2W，不能重复求逆。`--intrinsics-size WIDTH HEIGHT` 指内参对应图像尺寸，默认 518×392。其他输入结构沿用上游方法。

消融实验可加 `--no-pose-association`；保守模式使用 `--mode conservative --no-pose-association`，其默认阈值为 0.82。`instance_report --help` 提供网页报告入口。

计数图使用已有完整 Blender 场景及实例到网格名称的映射：

```bash
blender --background /path/to/scene.blend --threads 1 --disable-autoexec \
  --python-exit-code 1 --python bvb_v5/counting_views.py -- \
  --groups /path/to/instances.json --target table \
  --out /path/to/new/counting-views --local-views 4
```

映射示例：`{"groups":{"table":[["table-a"],["table-b-top","table-b-legs"]]},"room":{"ceiling":["ceiling"]}}`。计数图按输入实例映射着色，不会自动纠正错误的物体身份。

## 验证与限制

```bash
MPLCONFIGDIR=/tmp/bvb-mpl python -m unittest \
  bvb_v3.test_adaptive_peaks bvb_v5.test_selection \
  bvb_v5.test_dinov2_appearance bvb_v5.test_instance_sampling \
  bvb_v5.test_pose_association bvb_v5.test_v5_integration
```

这些是方法级测试，不需要模型、视频或 API key。Upstream provenance is recorded in UPSTREAM.json.


See ../RECONSTRUCTION.md for per-cluster reconstruction and shared-coordinate assembly.
