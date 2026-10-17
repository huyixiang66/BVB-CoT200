# BVB-CoT200：Blender 场景 → 八类 CoT → 单次答案

从已有 **BVB Astra / Luna xhigh `.blend`** 出发，运行固定的 200 道 VSI-Bench 问题（八类各 25 题、126 个场景）。场景来源与 QA 模型独立选择，**每题只请求一次，不做三次采样或投票。**

仓库包含代码、题目/评分标签、冻结的渲染配置与逐题 prompt；不包含 blend、原视频、渲染图片、API key 或历史实验目录。不是从视频重建场景；主 200 题协议不包含 debate 或场景微调。

主 200 题协议仍不包含场景微调。另提供独立的 [`refine.py`](refine.py) 研究入口，用于一题一题地比较原视频与 BVB 场景、生成受限几何候选并冻结前后 QA；它不会改变主评测结果。

## 环境

Python 3.11+、**Blender 5.2.2 LTS**（EEVEE 可运行）。Blender 使用自带 Python，不要用普通 Python 运行 `render_*.py`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export BLENDER_BIN='/path/to/blender'
export OPENAI_BASE_URL='https://your-provider.example/v1'
export OPENAI_MODEL='your-vision-model-id'
# 通过安全环境注入 OPENAI_API_KEY，不要把真实密钥写入仓库。
python run.py doctor
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`，环境变量使用 `$env:BLENDER_BIN='...'`。`.env.example` 只列出变量，程序不自动读取 `.env`。推荐 HTTPS；明文 HTTP 会暴露传输中的 key 和图片。

## 两个独立选择

| 参数 | 含义 |
| --- | --- |
| `--scene-model astra` / `--scene-model luna` | 选择 BVB **xhigh** 场景生成结果，默认 Astra |
| `--model MODEL_ID` / `OPENAI_MODEL` | 选择答题模型，命令行优先；`answer` 必须指定，无默认或自动回退 |

QA 服务须兼容 OpenAI Chat Completions 的图片输入和 `max_completion_tokens`。模型名称、图像数量与 token 限制以服务商为准；`xhigh` 仅描述已有场景的生成设置，不自动设置 QA reasoning effort。

## 输入与协议

| 题型 | 渲染与 CoT |
| --- | --- |
| 绝对距离、相对距离、物体尺寸、房间面积、路线规划 | 最新 C：固定 BVB 自带相机、16 帧逐题高亮、新 CoT；不加俯视图、米制网格或数字答案标注 |
| 计数 | 页面 v2：俯视/斜视/分区近景、网格、高亮与跨帧实例去重 CoT |
| 相对方向 | 页面 v2：站立视角、俯视网格、近景与观察者方向 CoT |
| 出现顺序 | 页面 v2：BVB 相机时间线 24 帧、高亮与首次可见帧排序 CoT |

另外三类只保留 v2 的视图与推理方法，**不保留投票**。Astra 保留冻结的逐题配置和 prompt；其中 q2583 无高亮、只有 6 帧，Astra 全量共 3,217 帧。

Luna 使用自身相机/时间线及物体名称，不套用 Astra 的手工物体名称。五类仍为固定相机 C 的 16 帧；另外三类根据实际匹配生成视图与说明，缺少高亮或站立/朝向参照时相应近景/站立视图会省略，实际帧数可能不同。Luna 是协议适配，不是与 Astra 完全相同输入的受控对比。

Astra 冻结文本在 [`data/prompts.json`](data/prompts.json)，配置在 [`data/recipes.json`](data/recipes.json)；Luna prompt 在渲染后生成。固定的是 BVB **估计出的**相机，不是原视频真实位姿；名称匹配也不等于实例真值。

## 运行

以 Luna 为例，先跑八题 smoke（每类一题）；换 Astra 就将各命令的 `luna` 改为 `astra`：

```bash
python run.py download --scene-model luna --smoke
python run.py render --scene-model luna --smoke
python run.py validate --scene-model luna --smoke
python run.py answer --scene-model luna --smoke --model your-vision-model-id
python run.py report --scene-model luna --smoke
```

全量运行去掉每条命令的 `--smoke`。默认资产目录为 `assets/<scene-model>/blends`，结果目录为 `runs/<scene-model>/smoke` 或 `runs/<scene-model>/main`。只测某题可加 `--only 720`；自定义目录用 `--run`，各阶段保持相同场景来源、题目选择和目录。

只有 `answer` 调用模型，发送题面、CoT 和渲染图片；默认每题最多 8,000 completion tokens（`--max-tokens`）、300 秒超时、无自动重试。`--workers` 只控制不同题目的并发数，不增加采样。更换 QA 模型请使用新的 `--run`，并在新目录执行前置渲染；不混用不同模型的答案。

下载固定 Hugging Face revision 并校验大小/SHA-256，Astra 约 2.42 GiB、Luna 约 0.58 GiB。国内可加 `--hf-endpoint https://hf-mirror.com`；需要认证时设置 `HF_TOKEN`，仅向指定下载端点发送，不转发给重定向目标。

已有文件按 `<scene_name>.blend` 命名，渲染加 `--blends /path/to/blends`，必须匹配所选来源的冻结哈希。网络或鉴权失败时可使用已有文件，不要绕过哈希检查。无需原视频。

Blender 串行运行，每进程两线程；不修改源 blend，不执行内嵌自动脚本。同一输出目录勿并发操作。成功结果校验后复用；来源/配置/源码改变应使用新目录，旧版运行目录不直接续用。

API 请求前写入 `requests.jsonl`，结束后写入 `answers.jsonl`。已记录成功或失败的题都不自动重发；如果中断发生在两者之间，程序会阻止续跑，避免对可能已计费的请求重复调用。先核查服务商记录与本地 PID，再处理该次运行；不要直接删除请求日志来“重试”。累计三次失败停止后续批次。

## 输出

| 文件 | 内容 |
| --- | --- |
| `renders/<qa_id>/prompt.txt`、`manifest.json` | 实际 prompt/哈希、帧路径/哈希、场景来源与未匹配角色 |
| `logs/`、`render_config.json` | Blender 日志与冻结配置 |
| `answer_config.json`、`requests.jsonl` | 模型、输入参数与请求开始记录 |
| `answers.jsonl` | 单次答案、原始响应、实际模型名、tokens、耗时、状态；不保存 key |
| `per_question.csv`、`summary.json`、`REPORT.md` | 逐题判分与八类汇总 |

只有 `report` 读取评分标签。采用历史 BVB pilot 的语义准确率规则（多数数值题容差 20%），**不是 VSI-Bench 官方 MRA，也不是 dual-VQA**。缺失/失败题保留在分母中。

本发布组合尚未跑出新的 200 题成绩，不能沿用旧 87/200 或 91/200；完整来源和局限见 [NOTICE.md](NOTICE.md)。

## 不花 API 费用的测试

```bash
python -m unittest discover -s tests -v
python tests/smoke_blender.py --blender /path/to/blender --scene-model astra
python tests/smoke_blender.py --blender /path/to/blender --scene-model luna
```

单元测试覆盖来源隔离、模型选择、动态 prompt、单次续跑和判分，API 为 mock。Blender smoke 使用合成几何测试 C、三类旧视图及 Luna 缺失匹配，仅检查元数据与哈希，不查看图片；不代表真实场景 QA 准确率。

主入口 `run.py`，调度/答题/报告在 `pipeline.py`，两套渲染器为 `render_fixed.py` 和 `render_legacy.py`。第三方许可见 `licenses/`，运行数据被 `.gitignore` 排除。

## 局部场景微调实验

### 当前开发基线：仅画面截断的多视角目标框评分

当前使用 `scripts/diagnose_box_multiview.py`，默认组合为
`--metric censored_edges --objective worst_gain`。不启用边缘分类，
不固定尺寸；旧背景/精确点方案保留为历史实验，不是本入口的搜索门禁。

```bash
python scripts/diagnose_box_multiview.py \
  --source /path/to/frozen-box-observations \
  --parts /path/to/verified-parts.json \
  --run runs/box-censored-new --blender /path/to/blender
```

输入是已冻结的目标框观测目录（`config.json` 中的 parent、`before_<pair_id>.json`
和 `calls/before_<pair_id>/request.json`，以及可校验的本地源帧/场景）；
不是从原始视频一键运行的入口。部件清单含 `case_id`、`blend_sha256`、
`body` 和 `carried`；`scripts/box_parts_q4697.json` 仅适用于对应开发场景。
私有观测、素材和运行目录不随仓库发布。

- 固定相机和时间配对，前三个有效视角拟合，第四个仅检查。
- 保留未裁剪 AABB 投影，画面截断边作为单向约束，其余边按坐标残差评分。
- 两轮搜索 XY 平移与统一缩放：径向平移上限 0.9 m、缩放范围 [0.7, 1.3]。
- 不调用模型或 QA、不读取 GT、不重渲、不保存或应用候选场景。
- `--edge-annotations` 和 `--translation-only` 仅为显式实验选项，当前基线不使用。

q4697 已知标签开发案例复跑得到 `[0, 0, 0.7]`，AABB 距离代理
**4.031000 -> 4.144850 m**；冻结后相对已知 GT 4.7 m 的绝对误差
从 0.669000 降到 0.555150 m。但第四帧评分仍退步，候选未应用，
不能宣称已建立可靠微调 pipeline。该方法由用户在已知开发结果后选回，
不是盲测方法选择。177 项单元测试通过；代码测试不代表真实几何改善。

### 历史入口与实验选项

新增独立入口 `refine_iterative.py`（实验性，尚未证明真实距离稳定改善）：

```bash
python refine_iterative.py --case /path/to/blind_case.json --run runs/evidence-new \
  --blender /path/to/blender --model your-vision-model-id --base-url your-configured-gateway \
  --preflight-only
# 预检通过后，保持所有参数不变，去掉 --preflight-only 执行。
```

- 不调用 QA，不读取标签。最多两轮；每轮只在未参与搜索的独立时间片验收通过后保存修改，失败保留上轮场景。
- 先以 AABB 连通性拆分名称候选组；多个组中仅有一个含题目字面名称时选该组，否则拒绝歧义。此规则不是实例真值。
- 用稀疏包围盒射线筛选源相机目标可见时间片，仍保留归一化时间对应，不跨时间寻找相似画面；零射线命中不等于严格不可见。
- 比较原材质、原画幅的重建。先核查原视频与重建中的目标身份和背景，再将合格帧分为搜索与验收；不强迫修改。
- 每轮候选含不改、靠近/远离和缩放，第二轮步长减半；缩放保持底部支点，不通过中心缩放改变落地高度。
- `outcome.json` 区分对象歧义、视角不足、配对失败、候选无改善、留出验收拒绝。`raw_group_gap_m` 到 `before_gap_m` 是实例分组量测变化，不能冒充场景修改增益；只比较同一分组的 `before_gap_m` 与 `after_gap_m`。
- 来源、代码和策略哈希冻结；更改实现必须使用新 run。旧 v4 入口及实验不覆盖。

当前版本支持逐题运行以及冻结题库中全部 **25 道 `object_abs_distance`** 的批量运行。批量 case 根据题目角色自动生成，不写入 GT、历史答案或诊断候选；所有题共享同一组预先冻结的对称候选：`no_change`、向目标靠近/远离 `0.15/0.30/0.45 m`、缩放 `0.85/1.15`。Blender 只按对象名称生成 mesh 候选组，不把它冒充为经过验证的物理实例。

批量生成需要分别提供原视频、BVB 视频和 `.blend` 目录。文件名模板支持 `{dataset}`、`{scene}` 和 `{qa_id}`；以下示例适用于原视频按数据集子目录保存为 `<dataset>/<scene>.mp4` 的目录：

```bash
python refine_batch.py generate \
  --original-videos /path/to/original-videos --original-template '{dataset}/{scene}.mp4' \
  --bvb-videos /path/to/bvb-videos --bvb-template '{scene}.mp4' \
  --blends assets/astra/blends

python refine_batch.py resolve
python refine_batch.py prepare
python refine_batch.py inspect
python refine_batch.py retrieve  --audit-model your-vision-model-id
python refine_batch.py candidates
python refine_batch.py select    --audit-model your-vision-model-id
python refine_batch.py finalize
python refine_batch.py answer    --qa-model your-qa-model-id --condition both --samples 1
python refine_batch.py report
```

各阶段结果默认写到 `runs/refine-object-abs-distance/items/q<id>/`，批次状态写入 `batch_<stage>.json`。可用 `--only 720 4740` 限定题目。批量入口不提供 candidate override：`finalize` 始终采用盲选结果，包含合法的 `no_change`。GT 只在 `report` 阶段从 `data/labels.jsonl` 读取。

单题仍可复制 [`data/refine_q4740.example.json`](data/refine_q4740.example.json)，填写路径并人工指定已核查的 mesh 名称/前缀。也可以只写 `{"label":"monitor","auto_match":true}`，先运行 `resolve` 冻结名称候选。所有阶段写入独立 `--run` 目录，源 `.blend` 只读，修改后的副本保存在 `final/scene_after.blend`。

```bash
export REFINE_AUDIT_MODEL='your-vision-model-id'
export OPENAI_MODEL='your-qa-model-id'

python refine.py doctor    --case my_case.json --run runs/refine-q4740
python refine.py resolve   --case my_case.json --run runs/refine-q4740
python refine.py prepare   --case my_case.json --run runs/refine-q4740
python refine.py inspect   --case my_case.json --run runs/refine-q4740
python refine.py retrieve  --case my_case.json --run runs/refine-q4740
python refine.py candidates --case my_case.json --run runs/refine-q4740
python refine.py select    --case my_case.json --run runs/refine-q4740
python refine.py finalize  --case my_case.json --run runs/refine-q4740
python refine.py answer    --case my_case.json --run runs/refine-q4740 --condition both --samples 1
python refine.py report    --case my_case.json --run runs/refine-q4740
```

`retrieve` 和 `select` 只能读取 `blind_case.json`，其中没有 GT 或历史答案。`retrieve` 在 8 组归一化时间配对上联合查找原视频/BVB 的稳定参照，不要求像素对齐。近似相机视角证据不足时允许并推荐选择 `no_change`。审计调用对 `gpt-6-*` 显式固定其唯一支持的 `temperature=1`，其余 QA 模型保持 `temperature=0`。候选渲染和最终前后 QA 统一使用 target-colour-only 高亮：moving object 为红色、reference object 为蓝色，其余可渲染 mesh 灰度化。默认 QA prompt 是与此高亮匹配的绝对距离 CoT，高细节输入与 `8000` completion token 上限，不再为无关尺度参照物添加第三种颜色。

时间配对使用 `normalized_time_pairs_v4`：两段视频按各自完整时间线的相同进度选帧，不使用相同帧号或播放秒数。候选仍由 BVB 自带相机重渲染；`select` 保留检索前两组 `pair_id`，按对应的 Blender 帧号查找每个候选，不再取排序后的首尾帧。`retrieval_pairs.json`、`candidate_spec.json` 和 `selection_pairs.json` 冻结对应关系；缺少对应帧或输入变化会在模型调用前停止。当前 BVB 视频帧号到 Blender 时间线按全程均匀采样插值，标记为近似映射，不能当作真实相机位姿或精确时间标定。

时间配对修复本身不改变候选动作或相机；最终 QA 帧数的后续调整见下文。请使用新的 `--run` 目录；旧检索/选择结果不会被覆盖，旧 `select_candidate` 调用缓存也不会作为 v4 结果复用。

自动名称解析只提供候选 mesh 组。若任一目标名称在 `.blend` 中无法解析，该题会自动冻结为 **`no_change`-only**，pipeline 仍可继续渲染和答题，但不允许猜测或移动其他 mesh。若名称存在但对应多个同类实例，则依靠原视频对齐阶段判断；证据不足时同样输出 `no_change`，不得用 GT 选择实例或候选。批量报告中的 `no_change` 是正式 pipeline 结果，但不表示几何得到改善。

若仅做使用 GT 的机制诊断，必须显式覆盖候选并留下标记。先在独立的 `runs/refine-q4740-diagnostic` 中完成上述 doctor 至 select 阶段，再执行以下命令；不能覆盖已经 finalize 的盲实验。以下仅是 q4740 将 monitor 向 bed 移动 0.30 m 的诊断示例，不是默认实验：

```bash
python refine.py finalize --case my_case.json --run runs/refine-q4740-diagnostic \
  --candidate-id toward_030 --diagnostic
python refine.py answer --case my_case.json --run runs/refine-q4740-diagnostic \
  --condition before --samples 1
python refine.py answer --case my_case.json --run runs/refine-q4740-diagnostic \
  --condition after --samples 3
python refine.py report --case my_case.json --run runs/refine-q4740-diagnostic
```

微调最终 QA 使用与输入 BVB 视频相同的帧数（例如 64 帧），沿 `.blend` 完整时间线均匀采样，全部送入 QA；不是固定 16 帧。逐帧记录并校验源相机的位姿与镜头参数，微调前后保持一致。没有原始采样 manifest 时，时间映射仍是均匀采样近似，不宣称与原视频真实相机位姿完全重合。主 200 题的 16 帧协议不变。

重复回答使用冻结的全部帧、提示和温度，逐次保存，**不投票、不自动重试**。更换帧数必须使用新 run，不能覆盖旧 16 帧 spec 或复用不同输入的 QA 缓存。单次答得更接近 GT 只说明模型输出存在可利用的波动；只有冻结输入下的结果分布稳定改善，才能支持微调有效。`report.json` 同时记录盲选结果、是否为 diagnostic、前后几何距离（当前为对象组 AABB 间距代理）、每次原始答案、MRA、均值和最佳值。

`no_change` 时 after QA 帧由 before QA 帧逐字节复制，manifest 必须记录 `qa_frames_byte_identical=true`；避免把两次独立重渲染的差异误当成几何微调效果。

新增测试：

```bash
python -m unittest tests.test_refinement tests.test_refine_batch -v
python tests/smoke_refine_blender.py --blender /path/to/blender
```

### Experimental Numeric Evidence Continuation

`refine_measured.py` continues a completed, blind, no-edit `refine_iterative.py`
run in a separate directory. It freezes single-pair target boxes and stationary
background points before scoring candidate Blender projections. It does not
read benchmark labels or call QA.

```bash
python refine_measured.py --parent runs/my-evidence-case --run runs/my-measured-case \
  --blender /path/to/blender --model your-audit-model --base-url your-api-base
python -m unittest tests.test_refine_measurement -v
blender --background --factory-startup --python-exit-code 1 \
  --python tests/blender_projection_checks.py
```

The fixed policy rejects missing/truncated targets, degenerate landmarks,
large background mismatch, and geometry/localization disagreement. A candidate
must lower normalized box residuals in both search views by at least 0.015,
then pass a fresh heldout numeric check and artifact audit. At most two rounds
reuse the existing shrinking-step candidate policy. Training measurements are
fixed, not repeatedly regenerated to favor a proposal.

This is an experimental proxy, not calibrated metric optimization: VLM boxes
and sparse projected AABBs can be wrong, and residual reduction does not prove
GT improvement. The first real q4697 continuation abstained on an unavailable
original target box; no successful real edit has been demonstrated. No retries,
no forced changes, no camera changes, and no changes to the main QA protocol.

`--measure-before-split` opts into `measurement_before_split_v2`. It accepts a
completed no-edit parent or a ready `--preflight-only` parent, measures all frozen
proposed pairs (at most eight), skips unusable measurements, and only then chooses
two training views plus temporally separated holdouts. Pair order, not error
magnitude or candidate scores, determines the split. No candidates are searched
until screening completes. Holdout coordinates are screened upfront but excluded
from search scores; this is not an entirely unseen holdout. Old run directories
must not be reused after code changes. This protocol has not demonstrated a
successful real edit or GT convergence.

`--measure-before-split --partial-features` opts into
`partial_surface_features_v1`. Cropping, occlusion, and incomplete reconstruction
do not by themselves reject a target. The audit model locates 2-6 corresponding
visible physical points plus stationary background landmarks, without requiring
complete-object boxes. Blender binds baseline points to first-hit surfaces of
the frozen moving mesh group, then projects the same local anchors under each
candidate. Wrong-object hits and lost/occluded anchors cannot improve the score.
The observed subset never shrinks the transform group. This remains a proxy:
ray binding does not prove semantic correspondence or complete instance membership.
Background and independent acceptance gates remain; no GT or QA is used.

Adding `--align-background` selects `partial_background_alignment_v2`.
It fits only stationary landmark coordinates, trying translation before a 2D
rotation/uniform-scale mapping. Every landmark is predicted by a fit excluding
that landmark. At least four distributed correspondences are required, with
bounded heldout error, scale, rotation and target extrapolation. The prompt asks
for 6-10 points when genuinely available. Target features never fit the mapping.
The inverse mapping expresses original target observations in baseline image
coordinates, leaving candidate geometry and source cameras unchanged.

This is a 2D nuisance correction, not physical camera recovery; depth parallax
and wrong correspondences can invalidate it. Existing q4697 coordinates provided
zero newly usable pairs in an offline replay because landmark counts/distribution
were insufficient. Synthetic tests demonstrate shared-motion removal and preserved
target residuals, not real-world GT improvement. Use fresh run directories; do not
rewrite old frozen hashes.
