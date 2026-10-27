# 包络视角选择

给定 BVB 的 Blender 场景和题目目标物体，按目标联合包络自动选择最少渲染视角。选视角阶段只读几何，不读像素：先在包络外采样候选相机，再做视锥可见性和遮挡射线检测，最后用贪心集合覆盖选出最小可读视角集合。该工具独立于 200 题主协议，不改变现有渲染和答题路径。

当前是单场景可复现原型，示例为 VSI-Bench `0d2ee665be` 的 32 道非 appearance-order 题。

## 流程

1. 题目 spec -> 目标物体
2. 目标物体 -> 联合包络 / 包围球
3. 包络球面 + 俯视候选视角
4. 视锥可见性 + first-hit 遮挡检测
5. 贪心集合覆盖 -> 最少可读视角
6. Blender 渲染选中视角

## 输入

- `scene.blend`: BVB 场景，由 Blender 命令行加载
- `spec.json`: 题目、题型和目标物体
- `object_groups.json`: 语义物体到 mesh 实例的映射

`spec.json` 只需要 `qa_id`、`question_type`、`question`、`targets`。示例保留的 `roles` 只用于人工核对，代码不依赖。

```json
{
  "scene": "0d2ee665be",
  "questions": [
    {
      "qa_id": "1747",
      "question_type": "object_abs_distance",
      "question": "Measuring from the closest point of each object, what is the distance between the keyboard and the cutting board (in meters)?",
      "targets": ["keyboard", "cutting board"]
    }
  ]
}
```

`object_groups.json` 中，一个实例是一个 mesh 名称列表；同一语义物体的多个实例放在外层列表里。计数题按实例分别检查可见性，避免把一个物体的多个 mesh 当成多个物体。

```json
{
  "groups": {
    "keyboard": [["black keyboard"]],
    "pan": [
      ["pan handle", "stacked frying pans"],
      ["pan handle.001", "stacked frying pans.003"]
    ]
  },
  "room": {
    "floor": ["main oak subfloor"],
    "ceiling": ["main ceiling"]
  }
}
```

## 快速开始

```bash
# 1. 选视角：只读几何，不渲染，不调用模型
"$BLENDER_BIN" --background --disable-autoexec /path/to/0d2ee665be.blend --threads 2 \
  --python-exit-code 1 --python scripts/envelope_views.py -- \
  --spec data/envelope_views/0d2ee665be/spec.json \
  --groups data/envelope_views/0d2ee665be/object_groups.json \
  --out runs/envelope_views/geometry.json

# 2. 渲染选中视角
"$BLENDER_BIN" --background --disable-autoexec /path/to/0d2ee665be.blend --threads 2 \
  --python-exit-code 1 --python scripts/render_envelope_views.py -- \
  --geometry runs/envelope_views/geometry.json \
  --out runs/envelope_views/renders \
  --manifest runs/envelope_views/render_manifest.json
```

两步必须使用同一个 `.blend` 场景。Windows 下使用 `$env:BLENDER_BIN='...'`。

## 参数

### 选视角

| 参数 | 作用 |
| --- | --- |
| `--spec spec.json` | 题目与目标物体 |
| `--groups object_groups.json` | 语义物体到 mesh 实例的映射 |
| `--out geometry.json` | 输出几何与选中视角 |
| `--summary geometry.md` | 输出逐题摘要，默认与 `--out` 同目录 |
| `--qa 1747,1748` | 只跑指定 QA id，逗号分隔 |
| `--scene name` | 覆盖 spec 中的场景名 |

### 渲染

| 参数 | 作用 |
| --- | --- |
| `--geometry geometry.json` | 上一步的几何输出 |
| `--out renders/` | 渲染根目录，每题一个子目录 |
| `--manifest render_manifest.json` | 输出渲染清单与文件哈希 |
| `--res 512` | 正方形分辨率，默认 512 |
| `--qa 1747,1748` | 只渲染指定 QA id，逗号分隔 |

## 输出

- `geometry.json`: 候选视角、可见性/遮挡指标、每题 `chosen_visibility` 和 `chosen_readable`
- `geometry.md`: 按题型的 K 分布和逐题选中视角
- `renders/<qa_id>/*.png`: 选中视角渲染图，不随仓库上传
- `render_manifest.json`: 渲染文件、相机参数、尺寸和 SHA-256

## 示例结果

`data/envelope_views/0d2ee665be/` 是最小输入示例。该场景 32 道非 appearance-order 题共渲染 47 张 512×512 图，平均每题 1.47 张：

| 题型 | n | 平均可读视角数 |
| --- | ---: | ---: |
| object_abs_distance | 7 | 1.29 |
| object_counting | 2 | 1.00 |
| object_rel_direction_hard | 3 | 2.00 |
| object_rel_direction_medium | 1 | 2.00 |
| object_rel_distance | 8 | 2.12 |
| object_size_estimation | 10 | 1.00 |
| room_size_estimation | 1 | 1.00 |

答题模型 `gpt-6-astra`、`reasoning_effort=high` 下，数值题官方 MRA 为 75.9%（20 题），选择题 MCA 为 91.7%（12 题）。32 题等权混合分为 81.8%，只作为本示例汇总，不作为官方总指标。这只是单场景结果，不是跨场景通用性结论。

主要失败是计数题把 pan 的多个叠放 mesh 当成多个物体，以及绝对距离缺少米制尺度参照。

## 当前限制

- `spec.json` 和 `object_groups.json` 是人工构造的适配层；尚未自动把自然语言题目 grounding 到 Blender 物体。
- 一个语义物体由多个 mesh 组成时，实例分组仍需要人工确认。
- 绝对距离和尺寸题缺少米制标尺；示例未加入网格或尺度参照。
- appearance-order 题不在覆盖范围内。
- 换场景可以复用几何算法，但需要提供该场景的 spec、object groups 和 room floor/ceiling 映射。

## 文件

```
scripts/envelope_views.py          # 包络采样、可见性/遮挡检测、贪心选视角
scripts/render_envelope_views.py   # 渲染选中视角
data/envelope_views/0d2ee665be/
  spec.json                        # 32 道题的输入示例
  object_groups.json               # 单场景物体实例映射示例
docs/envelope-view-selection.md    # 本说明
```
