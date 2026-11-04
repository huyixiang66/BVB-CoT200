# BVB 目标搜寻、区域聚类与重建对照（2026-11-04）

三块实验的代码：**目标搜寻**（换检测器／换提示词／放宽阈值）、**几何＋外观加权聚类**、**按 V5 流程做的三场景小规模重建对照**。

这里只放代码和预设，不含任何数据产物（检测缓存、`results.json`、`review.html`、`*.tgz`、渲染图都不在库内）。

## 与实验的对应关系

| 目录 | 实验 | 入口 | 主要输入 |
| --- | --- | --- | --- |
| `detector/` | 提高目标物体搜寻成功率 | `target-search/run_search.py`、`pilot_detect.py` | `cache/candidates.json`、`panel/`、`delta/` |
| `cluster/` | 固定波峰帧对比三种聚类 | `group_regions.py`、`group_peak_regions.py` | `verified/results.json`、`verified/peak-results.json` |
| `astra-pilot-20261104/` | 学长新重建流程三场景对照 | `prepare_pilot.py → stage_results.py → render_pilot.py → review_pilot.py` | `cluster/verified/results.json`、`senior/` |

## 1. 目标搜寻 `detector/`

| 文件 | 作用 |
| --- | --- |
| `analyze_video.py` | 建冻结帧缓存：GroundingDINO 检测 + Farneback 光流。写 `config.json`、`candidates.json` |
| `detect_cached_frames.py` | 在冻结帧上跑指定检测器，逐帧写 `detections.jsonl`，支持断点续跑 |
| `pilot_detect.py` | 成对 prompt／阈值试验，复用学长的后处理（短语均分、类别 argmax、NMS） |
| `mm_grounding_compat.py` | 给 Transformers 5.16.1 的 MM-GDINO 恢复逐层解码头张量，否则加载后预测头对不上 |
| `evaluate_pilot.py` | 人工参考框（冰箱 13 正例／12 负例，毯子 10 正例／12 负例）。只用于计分，不输入检测器 |
| `fridge-large-server-20261104/detect_large_audited.py` | large 变体的审计版 |
| `fridge-large-server-20261104/convert_mappings.py` | 把 MM 官方 checkpoint 转成 HF 权重 |
| `fridge-large-server-20261104/convert_local.py` | 转换的本地封装，读当前目录下的 `checkpoint-keys.json` |
| `target-search/run_search.py` | 补跑缺失的成对试验，复用已有冻结检测；读 `panel/`、`delta/` |
| `target-search/report_search.py` | 出指标与叠框图；要求 `search.log` 出现 `SEARCH_DONE` 才继续 |

固定权重版本：Base `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`，MM `b32293f50a56f7218d9f73ff940ffb9eba84d63c`。
阈值常量在 `pilot_detect.py:25`：`normal = (0.20, 0.25)`、`loose = (0.12, 0.17)`，这是短语 token 均值与最高 token 分，**不是后续波峰阈值**。

结论（同一冻结诊断集，13 正例／12 负例）：MM Base 原 20 类命中 11/13，Base＋`built-in refrigerator` 10/13，Tiny 原词表 5/13、放宽到 6/13，Base 原词表 0/13。收益主要来自换模型和加提示词，放宽阈值只找回个别漏框。

## 2. 聚类 `cluster/`

| 文件 | 作用 |
| --- | --- |
| `group_regions.py` | 核心：几何 g＋外观 s 加权，complete-link 合并 |
| `group_peak_regions.py <source> <out>` | V3 波峰不拆，只合并有支持的峰对 |
| `group_ray_regions.py` | 两条射线独立指向同一 ROI 时补一条几何链接 |
| `review_regions.py` | 生成局部审阅页 |
| `two-case-review-20261104/tune_door.py` | 2607 那题 36 帧的复跑 |

帧对分数 `g + 0.75 · s · (1 − g)`：`g` 来自 LingBot 位姿＋画面中央稀疏深度的区域接近度与表面重叠，`s` 来自同类别检测框扩框后裁图的 DINOv2 余弦相似度。预设（`preset/geometry-appearance-075.json`）：外观系数 0.75、外观生效的最低几何分 0、至少 1 对帧支持、合并门槛 0.468。

在 2607 上把 33／47／138 秒同一扇玻璃门的 `p02/p03/p06` 并成一组；在 2077 上把 `chair-p06/p07/p12` 并成一组；2077 全题时 `chair-p07/p12` 配对分 0.869 仍未合组，因为各自先与别的类别成组，complete-link 链上有低于 0.468 的边。

## 3. 重建 `astra-pilot-20261104/`

四步：`prepare_pilot.py` 冻结六题并按类别组织区域包 → Astra 子代理逐组输出 `scene-transport.json` → `stage_results.py` 按 V5 schema 校验并打 `server-inputs.tgz` → 服务器 Blender 编译、`render_pilot.py` 在同坐标拼合 → `review_pilot.py` 出原帧与重建对照页。

`reconstruct_regions.py` 是受限区域重建的提交入口（问题相关物体细建，周边用灰盒交代布局）。

## 目录假设：要跑起来需要改的四行

这几个脚本沿用原实验目录的相对结构，路径是写死的：

| 文件 | 行 | 常量 |
| --- | --- | --- |
| `astra-pilot-20261104/prepare_pilot.py` | 13 | `WORKSPACE = ROOT.parents[2]`（并从 `WORKSPACE/docs/experiments/` 读预设、从 `ROOT/cluster/verified/` 读聚类结果） |
| `reconstruct_regions.py` | 16 | `WORKSPACE = ROOT.parents[2]` |
| `reconstruct_regions.py` | 17 | `SENIOR = ROOT / "senior"`（学长 V5 源码） |
| `reconstruct_regions.py` | 18 | `CACHE = ROOT.parent / "code-selected-reconstruction-20261103" / "inputs"`（冻结输入缓存） |

把本目录放到 `<workspace>/tmp/<run>/` 且工作区正好在上两级，就能原样跑；否则改这四行常量。

## 运行依赖

- 检测：`torch`、`transformers`（Tiny／Base）、`safetensors`、`Pillow`、`numpy`；MM-GDINO 必须先过 `mm_grounding_compat`
- 聚类：`numpy`、`scipy`
- 渲染：Blender 4.x（`bpy`）
- 重建：OpenAI 兼容端点（Astra）、`senior/` 源码

## 复现边界

检测数字来自冻结诊断集，不是全视频召回率，也不是 QA 准确率。聚类在 2607 上调参、2077 上做迁移核查，保持实验状态。重建只跑了六题共 25 个区域组，未做 QA，跨组同一物体可能重复生成（1585 的沙发、2607 的多个木门假设都没额外去重）。
