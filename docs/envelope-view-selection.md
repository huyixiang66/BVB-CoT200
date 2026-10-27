# 包络视角选择

给定 BVB 的 Blender 场景和题目目标物体，按目标联合包络自动选择最少渲染视角。选视角阶段只读几何，不读像素：先在包络外采样候选相机，再做视锥可见性和遮挡射线检测，最后用贪心集合覆盖选出最小可读视角集合。该工具独立于 200 题主协议，不改变现有渲染和答题路径。

支持两种输入模式：

- **自动模式（推荐）**：从 `questions.jsonl`、`recipes.json` 和 `.blend` 自动生成题目目标、物体实例与房间区域。
- **文件模式**：继续读取人工确认过的 `spec.json` 和 `object_groups.json`，用于复现旧结果或人工修正疑难场景。

自动模式覆盖 6 类静态空间题：counting、absolute distance、relative distance、size、room size、relative direction。appearance order 和 route planning 会被明确跳过并写入 resolution，不会被静默当成成功。

## 流程

1. 题目 -> recipe roles / 精确 highlights / 保守名称匹配 -> 目标 mesh
2. 目标 mesh -> 语义实例 -> 联合包络 / 包围球
3. 地板/天花板连通区域 -> 每题房间与相机边界
4. 包络球面 + 俯视候选视角
5. 视锥可见性 + first-hit 遮挡检测
6. 贪心集合覆盖 -> 最少可读视角
7. Blender 渲染选中视角

## 输入

- `scene.blend`: BVB 场景，由 Blender 命令行加载
- 自动模式：`questions.jsonl`、`recipes.json`；场景名默认取 `.blend` 文件名，也可用 `--scene` 覆盖
- 文件模式：`spec.json`、`object_groups.json`

自动模式会在 `runs/envelope_views/<scene>/` 写出 `spec.json`、`object_groups.json` 和 `resolution.json`，便于审计每次匹配。目标解析顺序为：

1. recipe 的 `roles`；
2. recipe 的 `spec.highlights[*].match` 精确 mesh 名；
3. `name_resolver` 的保守名称匹配。

同一个 term 在整个场景内只建立一次实例表；若任一题目给出了精确 highlight，则该 term 优先使用精确 highlight 的并集，避免不同题把同一物体解析成不同集合。

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
# 自动输入 + 选视角：只读几何，不渲染，不调用模型
"$BLENDER_BIN" --background --disable-autoexec /path/to/scene.blend --threads 2 \
  --python-exit-code 1 --python scripts/envelope_views.py -- \
  --questions data/questions.jsonl \
  --recipes data/recipes.json \
  --qa 1723 \
  --out runs/envelope_views/geometry_1723.json

# 或：使用人工确认过的单场景输入
"$BLENDER_BIN" --background --disable-autoexec /path/to/0d2ee665be.blend --threads 2 \
  --python-exit-code 1 --python scripts/envelope_views.py -- \
  --spec data/envelope_views/0d2ee665be/spec.json \
  --groups data/envelope_views/0d2ee665be/object_groups.json \
  --out runs/envelope_views/geometry.json

# 渲染选中视角
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
| `--questions questions.jsonl` | 自动模式：题目文件，必须和 `--recipes` 同时提供 |
| `--recipes recipes.json` | 自动模式：recipe 文件 |
| `--scene-model astra|luna` | 自动模式：写入 resolution，用于标记场景来源，默认 `astra` |
| `--input-dir runs/...` | 自动模式：保存本次自动生成的 spec、groups、resolution |
| `--spec spec.json` | 文件模式：题目与目标物体 |
| `--groups object_groups.json` | 文件模式：语义物体到 mesh 实例的映射 |
| `--out geometry.json` | 输出几何与选中视角 |
| `--summary geometry.md` | 输出逐题摘要，默认与 `--out` 同目录 |
| `--qa 1747,1748` | 只跑指定 QA id，逗号分隔 |
| `--scene name` | 覆盖场景名；自动模式默认取 `.blend` 文件名 |

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
- 自动模式 `spec.json` / `object_groups.json` / `resolution.json`: 本次物体匹配、实例数、房间区域、跳过和未解析项
- `renders/<qa_id>/*.png`: 选中视角渲染图，不随仓库上传
- `render_manifest.json`: 渲染文件、相机参数、尺寸和 SHA-256

`geometry.json` 中的每道题还包含 `room.floor_objects`、`room.ceiling_objects`、`room.camera_bounds` 和 `room.selection`，因此多房间场景不需要共用同一个全局相机边界。顶层 `input_resolution` 保存自动输入审计信息。

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

- 自动 grounding 是 recipe/名称/AABB 规则，不是视觉模型；未命中的 term 会写入 `unresolved`，不会伪造目标。
- 多 mesh 物体的实例分组仍是启发式。计数题的房间范围、叠放物和同义类别可能和 GT 不一致；要做正式计数评测时仍需人工核对 `object_groups.json`。
- 地板/天花板按连通区域检测。不同房间的地板若在几何上直接相连，可能被合并成一个区域；`room_size_estimation` 默认使用所有检测到的房间区域。
- route planning 和 appearance order 不在覆盖范围内；它们不会被自动模式处理。
- 绝对距离和尺寸题缺少米制标尺；示例未加入网格或尺度参照。
- 这不是完整 video pipeline：不选视频帧、不用光流、不做 VGGT/相机轨迹补全；它只解决“已有 3D 场景下，按题选最少观察视角”。
- 可跨场景复用的是几何与自动输入流程，前提是 `.blend` 保留 BVB/Luna 的命名和 collection/parent 结构；任意第三方场景仍可能需要适配。

## 文件

```
scripts/envelope_views.py          # 包络采样、可见性/遮挡检测、贪心选视角
scripts/envelope_inputs.py         # 从 questions/recipes/blend 自动生成 spec 与 object groups
scripts/render_envelope_views.py   # 渲染选中视角
data/envelope_views/0d2ee665be/
  spec.json                        # 32 道题的输入示例
  object_groups.json               # 单场景物体实例映射示例
docs/envelope-view-selection.md    # 本说明
```
