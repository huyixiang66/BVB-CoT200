# BVB V6：区域聚类、位姿选帧与最简场景构建

版本 **6.0.0**。基于 [pose-clustering-counting 分支](https://github.com/huyixiang66/BVB-CoT200/tree/pose-clustering-counting)的固定提交 `9a9c0a0b5e9342b2ca7d82cfa747d884d48dd232`，独立于现有 `bvb_v5`。

来源与导出文件哈希见 [UPSTREAM.json](UPSTREAM.json)。此发布只包含可运行方法、测试、依赖和说明，不包含上游实验快照或本地结果。所有命令在 `methods/adaptive_frame_selection` 下运行，Python 可使用已安装依赖的环境。

## 流程与 V5 的区别

1. **检测与波峰候选**：复用已有问题解析、GroundingDINO 和 Farneback 候选接口。检测器可选原版 Tiny/Base 或 MM Base，普通后处理门槛是短语均值 0.20、最高 token 分 0.25。它们与波峰门槛是不同参数。上游没有证明某个模型或提示词适合所有类别；没有把 `cabinet`、`built-in refrigerator` 自动加到所有任务里。
2. **保留原波峰**：每个 V3 波峰的候选先作为整体参与聚类，不先拆散峰内帧。光流仍负责候选波峰附近的变化和帧互补，不替代后面的三维区域关联。
3. **三维区域几何**：由 LingBot 的 C2W 和稀疏点云估计画面中央区域。取最近三个、样本索引相差不超过 12 的点云帧；投影到中央区域后，用 32×24 网格保留各格最近表面。少于 12 个有效点则视为缺少几何证据。深度用中位数并过滤到 0.45～1.55 倍范围。
4. **帧对融合**：几何分 `g = 0.70·exp(-0.5·(d/0.45)²) + 0.30·overlap`，其中距离与重叠半径都按全候选池的中位深度归一化。外观分 `s` 是同类别扩框 DINOv2 特征的最大余弦相似度，截断到 [0,1]。最终 `score = g + 0.75·s·(1-g)`，对应 [固定预设](preset/geometry-appearance-075.json)。几何门槛为 0，因此即使 g=0，s=1 也能得到 0.75；不是“必须几何支持”的合并规则。
5. **按类别 complete-link 聚类**：默认每个类别单独处理，峰对至少一对帧支持，阈值 **0.468**。合并两个组时，所有跨组峰对都须过门槛，避免单链不断扩张。这里输出的是区域假设，不是确认的物体实例或计数结果。上游的跨类别区域试验也保留在快照中，但不是 V6 默认模式。
6. **限制重构帧数**：聚类后最多 **8 帧/组**，贪心优先补全尚未覆盖的类别，再最大化与已选帧的最小机位/朝向差异，以清晰度打破并列。常规 V3 每峰四帧，因此保留至少四帧；如果输入峰本身不足四帧，记录 `minimum_four_satisfied=false`，不伪造目标观测。不同组可以复用同一源帧。
7. **位姿条件最简重构**：每组只生成相关目标、直接相关白块与墙体的紧凑 JSON，基础资产由本地编译器构建。精确位姿单独保存；给 VLM 的位姿为三位小数压缩表示。默认统一到首相机原点、Z-up、LingBot 原生尺度；可传已有参考相机中心做一次全局 Sim3。不会每组独立重定坐标。
8. **编译和合并**：验证 JSON 后本地 Blender 生成场景、双轴 20° 斜俯视及共享坐标合并场景。合并保留原世界坐标，不再估计变换，不做跨组物体去重、墙体吸附或拓扑修复。

## 目录与适配

- `selection.py`：参数化输入、精确帧/位姿哈希核对、按类别聚类与预算选帧。
- `cluster/`：上游几何相似度及 complete-link 实现，仅修改包导入路径。
- `reconstruction.py`：上游原生统一坐标、区域 prompt、选帧和单次 API 提交方法；改为显式 provider/model 参数并增加输入防错。
- `detector/`：上游冻结帧检测工具和 MM 解码头兼容修复；需先准备权重缓存。
- `runtime/`：固定的本地 V5 资产、schema、位姿压缩、预览和合并代码，补齐上游缺失的 `senior/` 和 `build_session_blender` 依赖。
- `build.py`：将原始模型 JSON 或已验证的 scene-spec 编译为 `.blend`，记录缺失或空响应。

没有把上游特定冰箱时段的人工补帧、诊断集标注或手工识别结果注入默认选帧。当前 `select` 从已经生成的 V3 波峰及 DINOv2 缓存开始，不声称一次命令就能从任意原视频完成所有步骤。

## 安装与验证

以下均在项目根目录执行，使用 Python 3.11+：

```bash
.venv/bin/python -m pip install -r bvb_v6/requirements.txt
.venv/bin/python -m bvb_v6 --help
.venv/bin/python -m unittest bvb_v6.test_pipeline
```

新跑检测/外观推理额外安装 `requirements-gpu.txt`。服务器下载使用 `HF_ENDPOINT=https://hf-mirror.com`。已有缓存可直接复用，选帧本身不调用模型或 API。

## 1. 选帧

```bash
.venv/bin/python -m bvb_v6 select \
  --v3 /path/to/v3/report \
  --cache /path/to/frame-cache \
  --poses /path/to/lingbot \
  --appearance /path/to/expanded-dinov2 \
  --qid 1585 --frame-cap 8 \
  --out /path/to/new/selection
```

省略 `--qid` 可处理题目文件中全部题目。输入分别包含：V3 的 `questions.json`、`peaks.json`；缓存的 `candidates.json` 与帧文件；LingBot 的 `camera-poses.npz`、`sampling.json`、`sparse-geometry-corrected.npz`、可选 `geometry-info.json`；外观的 `rois.json`、`features.npz`。必须使用正确 OpenCV C2W，不能重复求逆。

输出 `selection.json` 包含每题分组、峰对证据、所有候选、最终帧和缺失目标。默认缺少 ROI 特征会明确失败，修复外观缓存后重试；只有显式 `--allow-missing-appearance` 才将缺失外观视为没有外观证据，并记录缺失清单。无目标词的全局题需要提供独立场景上下文，不会凭空生成物体组。

## 2. 准备最简重构输入

```bash
.venv/bin/python -m bvb_v6 prepare \
  --selection /path/to/selection/selection.json --qid 1585 \
  --cache /path/to/frame-cache --poses /path/to/lingbot \
  --out /path/to/new/prepared
```

可选 `--reference-cameras /path/to/centers.json`，每条包含 `t` 与 `p`；至少三个非退化、时间匹配的相机中心。当前资产协议只支持 640×480 输入，其他分辨率会明确拒绝。准备不读取密钥，不调用 API，输出逐帧精确位姿、压缩位姿及 prompt。

## 3. 显式提交模型

```bash
.venv/bin/python -m bvb_v6 submit --out /path/to/prepared \
  --provider openai --model gpt-6-astra --reasoning high \
  --max-tokens 3600 --limit 1
```

本地默认沿用用户选择的官方 API 和 high；上游受限请求脚本默认 low，两者不应视为同一推理配置。官方接口读取 `LLM_API_KEY`，Boyue 需显式 `--provider boyue`，读取 `BOYUE_API_KEY` / `BOYUE_BASE_URL`。可用 `--key-env` 选择其他变量，密钥不写入产物。没有自动从官方切换到 Boyue。

默认串行、每次最多一个新请求；`--limit` 可显式增大。已有请求账本或有效结果不自动重复调用，失败立即停止。输出限额包含推理与可见输出，不保证有效 JSON。本入口按上游区域策略每组单次提交；**没有自动付费修复循环**。如需旧版编译错误反馈闭环，仍使用原 V5 入口，不混淆两套停止规则。

也可由外部流程把未经编译的完整 JSON 保存为各组的 `scene-transport.json`，再执行下一步。

## 4. 本地编译、预览、合并

```bash
.venv/bin/python -m bvb_v6 build \
  --source /path/to/prepared --out /path/to/new/build \
  --blender /path/to/blender
```

输出 `fragments/` 下各组场景与预览、`merged/merged-scene.blend`、俯视/斜俯视图和 `build.json`。缺失响应、空场景、未重构出目标都会记录；编译成功不等于回答正确。完整视频、原始结果和 V5 文件均不会被此入口覆盖。


## 最新策略：物体框聚类、无框语义审核和覆盖保留

`select` 保留原 V6 中央区域聚类作基线。最新路径继续运行物体框聚类：框内近表面点云、几何可靠性与 DINOv2 外观联合，缺失几何不算冲突；组间采用平均关联与支持比例，并检查同帧分离/可靠三维冲突。融合分门槛 0.468，外观权重 0.75；当前设置允许一对有效帧支持合并，使用 `--minimum-frame-pairs 1`（兼容默认值为 2）。聚类后按位姿互补选 4–8 帧，不按波峰数无限增加。

```bash
python scripts/run_v6_object_clustering.py --source /path/to/selection/selection.json \
  --minimum-frame-pairs 1 --out /path/to/object-clusters
python scripts/report_v6_object_clustering.py --source /path/to/object-clusters \
  --out /path/to/gallery --video /path/to/video.mp4 --detector-label 'MM Grounding DINO Base'
python scripts/validate_v6_object_clustering.py --source /path/to/object-clusters --delivery /path/to/gallery
```

图像报告保留全部候选的类别色框、检测分数与选中标记。这里的框用于人工展示；Astra 审核接收完全无框原帧和无框上下文裁剪，候选位置仅以文字坐标提供。每峰最高分有效框作为代表，按完整输入哈希跨题复用审核。检测分数与 GT 不发送给 Astra。

```bash
python -m bvb_v6.semantic_review prepare --selection /path/to/object-clusters/selection.json \
  --out /path/to/review --arm roi_robust
python -m bvb_v6.semantic_review submit --out /path/to/review
python -m bvb_v6.coverage prepare --source /path/to/review
python -m bvb_v6.coverage submit --source /path/to/review
python scripts/report_semantic_review.py --source /path/to/review --delivery /path/to/gallery
```

请在本地 `.env` 配置 `LLM_API_KEY`，也可以使用环境变量。审核和对比使用官方 API，模型 `gpt-6-astra`、high；只有 submit 阶段产生 API 请求。不要提交凭据。**当前审核 prompt 的 keyboard 消歧固定为 musical keyboard/piano，沿用本场景解释；用于其他场景前应根据语义调整并重新 prepare，旧请求哈希不会通用。**

语义聚合：任一代表证据 match 就 accept；全部明确误检才 reject；其他为 uncertain。随后独立决定是否选用：

- 有 accept：保留所有 accept。
- 某类唯一聚类：保留，标记 `only_candidate`。
- 多组但无 accept：优先让 Astra 对比 uncertain；没有 uncertain 则比较全部候选，保留相对最佳组，标记 `comparison_best`。
- 实际没有候选：明确记录缺失，不能凭空保证检测到目标；该题 prepare 被阻止。

原始判断保留，补选不等于确认物体身份。排序须完整、赢家须位于候选中；无有效比较结果不会自动编造赢家。每题每类的覆盖统计与选择记录输出到 `coverage-summary.json` 和 `reviewed-selection.json`。

```bash
python -m bvb_v6 prepare --selection /path/to/review/reviewed-selection.json --qid QUESTION_ID \
  --cache /path/to/frame-cache --poses /path/to/lingbot --out /path/to/prepared
```

审核后自动使用 `semantic_selected`，禁止绕过审核文件的指定 arm。prepare 保存准入回执；submit/build 检查准入、对比结果和输入帧哈希。补选组的重构 prompt 明确保留不确定性，不得为满足类别标签编造几何。代码不会自动恢复或提交重构。

审核、比较、检测和 QA 的分数不能互相当作准确率。该方法不保证消除误检，也不保证跨组物体唯一性；共享位姿合并不做新的 ICP、墙体吸附或去重。

## 验证

```bash
python -m unittest bvb_v6.test_pipeline bvb_v6.test_object_clustering \
  bvb_v6.test_semantic_review bvb_v6.test_coverage
```

测试覆盖聚类支持、干净输入像素、语义聚合、唯一候选保留、候选排序、比较结果篡改和重构门禁。导出版本另做 CLI、报告链接与离线 prepare 验证；不附带任何实验图片、视频、模型响应或结果 JSON。
