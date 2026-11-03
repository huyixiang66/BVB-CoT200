# 扩框、位姿归并与计数视角交接

该分支从学长的自适应选帧提交 `0cc1ffce` 接入两个模块。

## 聚类

入口和输入契约见 [自适应选帧说明](../methods/adaptive_frame_selection/README.md)。
更新的实现位于 `bvb_v4/pose_association.py`，通过
`instance_sampling --pose-association` 接入。输出沿用原来的组和选帧结构，
重建端可以继续把组号、原框、原帧和相机位姿一起交给 Astra，组号作为候选关联提示。

在 `3db0a1c8f3` 的三个开发题上，沙发的三个候选组并成一个，160/166 秒的鼠标
观测合并，138 秒玻璃门观测并入之前的玻璃门组。迁移后的 CLI 与实验缓存选帧一致。

| 题号 | 原自动组数 / 帧数 | 更新后 |
|---|---|---|
| 1585 | 5 / 24 | 3 / 16 |
| 1754 | 4 / 16 | 3 / 12 |
| 2607 | 5 / 26 | 4 / 23 |

候选检测错误可能继续存在。框内稀疏点也可能落在背景上。几何结果和局部特征
仍应保留供复核。原 V3/V4 的固定采样和图像尺寸假设见自适应选帧说明。

## 计数图

输入已有 Blender 场景和语义实例到网格的映射，每个实例对应一个网格名列表。
输出一张全局图、最多四张局部图和 `manifest.json`。同一实例的所有部件保持
固定颜色。供 VLM 使用的图由 `vlm_images` 列出，图内只包含场景颜色。

映射结构沿用选视角脚本的结构。

```json
{
  "groups": {"table": [["desk-top-a"], ["desk-top-b"]]},
  "room": {"floor": ["floor"], "ceiling": ["ceiling"]}
}
```

从仓库根目录执行以下命令。输出目录应尚不存在。

```bash
OMP_NUM_THREADS=1 LP_NUM_THREADS=1 blender --background scene.blend \
  --threads 1 --disable-autoexec --python-exit-code 1 \
  --python scripts/counting_views.py -- \
  --groups instances.json --target table --out counting-views \
  --global-view top --local-views 4 --resolution 1024
```

局部图按目标中心排序后分区。可用 `--axis y` 改变分区方向，也可以用
`--split 1.2 4.8 7.5` 给出 x 方向的三个区域边界，产生最多四个区域。
这是 578 办公室的验证配置。它依赖场景坐标，其他场景默认用等数量分区。
`--global-view oblique` 可改为斜视全局。需要隐藏顶部构件时设置
`--hide-above 2.30`，单位与 Blender 坐标一致。脚本默认隐藏映射里的天花板。

局部图通过目标包围盒解析构图，和全局图一起保留空间上下文。
内部映射仍需要检查是否把一个物体拆成多个实例，颜色会继承输入映射。

### 已有计数试验

578 场景的 2619 问桌子数量。旧两图回答 12，MRA 为 27.27%。
彩色全局加四张局部图回答 18，标准答案 19，MRA 为 90.91%。
本次输入只有颜色，没有图内编号，使用 API3 Astra high。
该试验同时改变颜色、视角数与渲染，记录为组合效果。现有 Blender 的桌面映射
含 20 个候选，和标准答案存在差异。

迁移后的计数 CLI 已在同一 Blender 上实际渲染，输出 20 个实例的颜色映射
和 5 张图。该检查验证入口与产物，计数回答来自前述实验版本。
