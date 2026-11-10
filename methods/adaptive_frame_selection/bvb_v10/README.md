# BVB V10 — 从原视频到问题场景和 QA

V10 冻结了本地 V9 9.2.0 的**联合重构**策略。所有保留聚类的原帧和共同 LingBot 位姿一起输入 Astra，生成一个统一场景；不再把多个独立重构的房间事后直接叠加。

本分支只交付方法、入口、依赖、测试和说明，不包含视频、实验图片、答案、场景文件、权重或凭据。本地十题实验仍保留原记录；本分支的调整主要是路径、参数和流程串接的可移植化。

## 流程与参数

1. **问题实体**：只读取公开题干、全部选项和题型。已有 `objects/roles` 原样使用；缺少时用 Astra 提取物体及角色。GT、历史回答、标注实例 ID 不进入模型。
2. **视频特征**：2fps 候选原帧；MM-Grounding-DINO Base 初始 peak-token 阈值 0.25、phrase-mean 阈值 0.20、类别内 NMS 0.5。类别并集可共享检测缓存，再按题目取子集；长词表按240 token分块。不同 caption 的分数不视为严格可比。
3. **几何与运动**：LingBot-MAP streaming，518输入、滑窗24、scale frames 4、SDPA，输出 C2W/内参及稀疏深度点。Farnebäck 对原视频相邻帧计算，缩放320×240，记录光流幅度P75；便携入口使用CPU，与原生CUDA工具保持同一指标，但不声称数值完全相同。
4. **逐题抽帧及聚类**：0.5波峰门槛，每峰4帧；光流挑邻帧，DINOv2外观与LingBot ROI几何联合聚类，至少两对独立帧支持，每组最多8帧。不使用SAM2。完整聚类实现保留在 `bvb_v6/object_clustering.py` 和 `scripts/run_v6_object_clustering.py`。
5. **Astra审核**：先按问题判断单/多实例。清图审核后，只要包含真实目标即可接受；有候选的类别不能全部拒绝，uncertain通过对比择优保底。单实例问题每类别只保留一组；计数、远近比较等可保留多组。无候选时明确报缺失，不伪造。
6. **联合重构（第0轮）**：所有保留组的去重原图、bbox文字观测及压缩位姿共同输入 Astra high，输出完整bpy代码。问题目标保留可辨识结构；其他物体只保留直接支撑/直接接触目标的白块。所有输入有证据的墙、门洞边界、地板都保留；门洞保持空隙；不补未见房间，不加屋顶或间接支撑链。
7. **执行与检查**：隔离运行Blender，检查目标实例上限、非目标白色长方体、直接支撑/接触引用和结构证据清单。精确LingBot相机后附到场景中，投影检查不改变网格。位姿条件生成不是数值ICP配准；声明通过也不保证几何准确。
8. **选视角及QA**：世界Z轴俯视＋八个双轴20°斜视图；Astra选其中一张，再用该单图独立回答选项题。没有GT输入。结果网页逐题更新，包含输入聚类、全场景、选中视角、答案及blend/代码下载。

默认 `gpt-6-astra / high / subscription_direct`，只跑第0轮，不自动启动迭代。默认不设硬输出token上限。`iterate`保留为单独实验命令，不属于默认流程。失败保留请求记录，不盲目重试；完成请求可按指纹复用，输入变化必须使用新目录。

## 环境

从仓库根目录进入：

```bash
cd methods/adaptive_frame_selection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-v10.txt
# GPU机器另外安装匹配驱动的PyTorch，并使用：
pip install -r requirements-gpu.txt
python -m bvb_v10 verify
```

Python>=3.11；GPU阶段需要CUDA。MM-DINO使用现有 `mm_grounding_compat.py` 恢复非共享decoder heads并逐张量核对权重，不替换检测模型。V10当前几何入口要求原视频640×480；其他尺寸应在所有检测和位姿步骤之前统一规范化。

LingBot代码及依赖按其官方仓库安装到 `INSTALL/lingbot-map/`，使 `from demo import load_images,load_model,postprocess` 可用；权重位于 `INSTALL/models/lingbot-map.pt`。然后：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m bvb_v10.download_models --lingbot-root /path/to/INSTALL
```

下载脚本固定MM-DINO、DINOv2、LingBot权重revision，并保存收据。它不安装LingBot源码，不修改CUDA驱动。现有GPU环境可以直接复用。

**Blender构建适配器当前使用macOS `sandbox-exec`**，拒绝网络访问，并只允许生成代码读取/写入自己的运行目录。Linux适合运行特征和聚类；构建在Mac继续，或自行提供等价的隔离Blender运行器。不能通过删掉隔离检查来假装已支持Linux构建。

第三种API的OAuth方法也随分支提供：

```bash
python scripts/test_subscription_direct.py
```

在本机完成独立授权；凭据仅写入被忽略的 `.local/siwc-direct/`，不上传Git。不要把API key当作OAuth token，也不会自动回退付费API。账户须能访问指定模型。传输适配器支持独立请求和token收据；缓存命中由服务端决定。

## 从原视频开始

输入JSON可以是列表或 `{"questions":[...]}`，每题包含：`id, scene_name, question, question_type, options`。已有解析结果可另带 `objects, roles`；一个文件对应一个视频，题号唯一。不要放GT。

```bash
# 1. 保留已有实体；缺少时LLM提取（会调用API）
python -m bvb_v10.workflow entities --questions /data/questions.json --out /runs/demo/entities

# 2. 原帧、检测、LingBot、原帧光流（GPU机器）
python -m bvb_v10.workflow features \
  --video /data/video.mp4 --questions /runs/demo/entities/questions.json \
  --out /runs/demo/features --lingbot-root /path/to/INSTALL
# 可单独运行：--stage frames|dino|poses|flow

# 3. 抽帧→聚类→审核→联合重构→渲染→选视角→QA→报告
python -m bvb_v10.workflow run \
  --questions /runs/demo/entities/questions.json --features /runs/demo/features \
  --out /runs/demo/v10 --device cpu \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

`run --device`控制DINOv2特征提取；Mac可用CPU，CUDA机器可用cuda。远程特征搬到Mac时用 `rsync -aL` 解引用帧软链接，保留帧哈希、索引、配置和npz不变；在目的机器重新运行抽帧和准备步骤，以生成正确路径。也可在GPU机器先运行下方独立 `sample`、`audit`，再按原目录结构挂载输出；不要手工篡改已冻结输入的哈希。

网页：`/runs/demo/v10/report/index.html`。仅刷新或添加展示用GT：

```bash
python -m bvb_v10.workflow report --out /runs/demo/v10
python -m bvb_v10.workflow report --out /runs/demo/v10 --reference /data/display-only-gt.json
```

GT文件格式为 `{"qid":"A"}`，仅由报告器读取；不影响重构、选视角、回答。运行过程根据已存在的收据恢复；无有效收据的半成品需要检查后处理，不能把中断请求算成功。

## 单阶段与已有缓存

```bash
python -m bvb_v10 sample --cache /data/features/detector/local-cache \
  --poses /data/features/poses --flow /data/features/native-flow.csv \
  --questions /data/entities/questions.json --out /runs/sampling --device cuda
python -m bvb_v10 audit --selection /runs/sampling/clusters/selection.json --out /runs/audit
python -m bvb_v10 prepare-regions --selection /runs/audit/input/review/reviewed-selection.json \
  --qid YOUR_ID --cache /data/features/detector/local-cache --poses /data/features/poses --out /runs/prepared
python -m bvb_v10 prepare --prepared /runs/prepared --out /runs/question
python -m bvb_v10 infer --out /runs/question
python -m bvb_v10 build --out /runs/question
python -m bvb_v10 render --out /runs/question
python -m bvb_v10.workflow qa --out /runs/question
python -m bvb_v10.workflow --help
```

当前QA适配器对应本次静态物体空间**选择题**。全局房间、路线、出现顺序、运动或数值尺度题仍需显式补充证据/评分适配；不会把缺少这些步骤的结果伪称完整支持。实体提取明确发现区域或时序要求时会停下提示。

## 验证

```bash
python scripts/test_bvb_v10.py
python -m unittest bvb_v10.test_portable
python -m bvb_v10 verify
```

锁文件覆盖V10及其本地依赖；源码变化需审查后重新冻结。离线测试覆盖公共字段去GT、单视频约束、冻结参数、构建契约及QA恢复。发布验证不重新调用付费/订阅模型，不把本地十题结果作为远程新复现实验。
