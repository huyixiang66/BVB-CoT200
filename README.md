# BVB-CoT200：Blender 场景 → 八类 CoT → 单次答案

## 纯几何选视角（实验入口）

仓库提供单题 Blender 几何选位脚本：输入 `.blend`、来源视频和题目 JSON，输出选中的相机位姿及 Blender 渲染图。运行时不调用 AI。当前版本只用 `.blend` 几何和题目；视频只记录在结果来源信息中，不参与选位。题目 JSON 需要显式提供题型及 Blender 网格名映射。详见 [几何选视角说明](docs/geometry-only-view-selector.md)。

从已有 **BVB Astra / Luna xhigh `.blend`** 出发，运行固定的 200 道 VSI-Bench 问题（八类各 25 题、126 个场景）。场景来源与 QA 模型独立选择，**每题只请求一次，不做三次采样或投票。**

仓库包含代码、题目/评分标签、冻结的渲染配置与逐题 prompt；不包含 blend、原视频、渲染图片、API key 或历史实验目录。不是从视频重建场景；主 200 题协议不包含 debate 或场景微调。

主 200 题协议仍不包含场景微调。独立的 [GPT 双帧反馈闭环](#gpt-双帧反馈微调) 允许模型逐轮调整相机、物体或同时调整两者；它不会改变主评测结果。

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

## GPT 双帧反馈微调

当前推荐流程是**多帧、共享 3D 修正**：把多组时间匹配的原视频帧和 BVB 渲染帧一起交给模型，并附上各帧的相机轨迹相对姿态；模型一次判断一个跨视角一致的修正，程序修改 Blender 场景副本、重渲全部视角，再把新结果送入下一轮。每轮可以调整整条相机轨迹、目标物体，或两者。不画框，不做候选排序，也不要求背景配准。`no_change` 只表示本轮不编辑，仍继续到 `--rounds` 上限；GT 不参与修正。

使用上面的 Python 环境与 `OPENAI_API_KEY`、`OPENAI_MODEL`、`OPENAI_BASE_URL`、`BLENDER_BIN` 配置。模型需支持图片输入、JSON 输出与 Chat Completions；代码不固定服务商或模型名称。

**准备输入。** 复制 `data/visual_feedback.example.json` 为自己的 `case.json`，修改 `inputs.blend` 为 `.blend` 绝对路径，修改 `objects.moving` 中的目标名称。`names` 精确匹配 Blender 网格名，`prefixes` 匹配名称前缀；同一物体的所有部件一起移动和缩放。

先准备多组一一对应的原视频/BVB 帧。`scene_frame` 是 BVB `.blend` 时间线中与该渲染帧对应的帧号；原视频帧与 BVB 渲染帧须属于同一时刻。下面的 manifest 只记录路径和时间对应，不包含图片：

```json
[
  {"pair_id": 2, "bvb_frame_index": 44, "source_frame_index_0based": 2144,
   "scene_frame": 1717, "source_frame": "/data/source_2144.jpg",
   "render_frame": "/data/bvb_44.png", "width": 512, "height": 512},
  {"pair_id": 3, "bvb_frame_index": 56, "source_frame_index_0based": 2729,
   "scene_frame": 2185, "source_frame": "/data/source_2729.jpg",
   "render_frame": "/data/bvb_56.png", "width": 512, "height": 512},
  {"pair_id": 5, "bvb_frame_index": 63, "source_frame_index_0based": 3070,
   "scene_frame": 2458, "source_frame": "/data/source_3070.jpg",
   "render_frame": "/data/bvb_63.png", "width": 512, "height": 512}
]
```

用同一个 BVB blend 准备所有配对。最后一组的相机姿态作为固定坐标参考；准备脚本同时记录每组相机的姿态和投影参数：

```bash
"$BLENDER_BIN" --background --disable-autoexec /absolute/path/scene.blend --threads 1 \
  --python-exit-code 1 --python scripts/prepare_visual_feedback.py -- \
  --case /absolute/path/case.json --pairs-manifest /absolute/path/pairs.json \
  --output runs/feedback-input-001
```

**运行多帧微调。** `--pair-ids` 按 manifest 中的 `pair_id` 选择训练视角。每轮模型看到所选所有原图/渲染图；程序对相机轨迹或目标应用一个共享 3D 修正，并重新渲染每组视角：

```bash
python scripts/run_visual_feedback.py --case runs/feedback-input-001/case.json \
  --pairs runs/feedback-input-001/pairs.json --pair-ids 2,3,5 --run runs/feedback-001 \
  --blender "$BLENDER_BIN" --model "$OPENAI_MODEL" --base-url "$OPENAI_BASE_URL" --rounds 5
```

模型动作中的平移使用固定参考相机的 +X 右、+Y 上、+Z 后退坐标，单位为 Blender 场景单位；相机修正作用于整条动画轨迹，目标修正作用于目标网格组。平移步长和缩放范围由程序校验。`--rounds` 为 1–5 次判断，默认 2；`no_change` 轮不渲染，但会继续下一次判断。每轮只从上一轮的场景副本继续；原 `.blend` 不覆盖。输出的 `edit_XX/render_pair_<id>.png` 是对应视角的更新渲染。

多帧模式要求 BVB 场景有可采样的活动相机；相机动画必须是无父级/约束、无 NLA/driver 的普通关键帧动画，目标网格不能带动画或约束。初始帧配对质量由用户提供的 manifest 决定；模型不会自动搜索对应帧。

**单帧兼容流程。** 只有一组配对时仍可用已有方式，使用 Blender 当前活动相机准备一个渲染帧，再运行 `--pair-id`：

选好一张原视频帧，使用 Blender 当前相机在对应场景状态生成初始渲染及可运行配置：

```bash
"$BLENDER_BIN" --background --disable-autoexec /absolute/path/scene.blend --threads 1 \
  --python-exit-code 1 --python scripts/prepare_visual_feedback.py -- \
  --case /absolute/path/case.json --source-frame /absolute/path/source_frame.jpg \
  --scene-frame 826 --output runs/feedback-input-001 --width 640 --height 480
```

`--scene-frame` 是 Blender 时间线帧号，不是原视频帧号；请使用对应的场景状态。准备阶段沿用该时刻的活动透视相机，不自动恢复原视频相机。如果已有调整更准的相机场景，可直接以该副本准备输入。`--case` 中的 blend 路径须与命令加载的文件一致。准备过程不调用模型，输出 `case.json`、`pairs.json`、`scene.blend` 和 `render.png`。

每轮使用上一轮修正后的场景和渲染帧，原视频参考帧保持不变：

```bash
python scripts/run_visual_feedback.py --case runs/feedback-input-001/case.json \
  --pairs runs/feedback-input-001/pairs.json --pair-id 0 --run runs/feedback-001 \
  --blender "$BLENDER_BIN" --model "$OPENAI_MODEL" --base-url "$OPENAI_BASE_URL" --rounds 5
```

`--rounds` 为 1–5，默认 2；单帧模式下模型可用 `stop` 提前结束。每次准备或运行都使用新的输出目录。已有 `case.json` 和 `pairs.json` 的用户可跳过准备阶段，并使用其中的 `pair_id`。

模型可返回 `subject=camera`、`target` 或 `both`。例如，同一轮联合修正：

```json
{
  "action": "adjust",
  "subject": "both",
  "camera": {
    "translation_local": [0, 0, 0.2],
    "rotation_local_degrees": [0, 2, 0]
  },
  "target": {
    "translation_local": [0.05, 0, 0],
    "scale_ratio": 1.05
  },
  "reason": "Brief explanation of the visual mismatch"
}
```

两者都使用**本轮修改前**的相机局部坐标：+X 画面右、+Y 上、+Z 后退。平移是 Blender 单位，不能直接称厘米。相机每轮平移长度不超过 0.5、各轴旋转不超过 8°；目标平移长度不超过 0.25、统一缩放为 0.8–1.25，围绕底部中心缩放。固定参照物和其他场景物体保持不变。

每轮结果在 `edit_01/` 等目录。多帧模式包含 `scene.blend`、每个视角的 `render_pair_<id>.png`、`spec.json`、`state.json` 与日志；单帧模式使用 `render.png`。`result.json` 记录最终场景、配对、各轮动作与应用状态；`calls/` 保存请求及响应。源 `.blend` 不覆盖，Blender 使用 CPU 单线程。需要继续时，把最终场景作为新的 `case.inputs.blend`，再准备输入并使用新的运行目录。

离线检查（不请求模型，也不读取图片像素）：

```bash
python -m unittest tests.test_visual_feedback
"$BLENDER_BIN" --background --factory-startup --threads 1 --python-exit-code 1 \
  --python tests/blender_visual_feedback_checks.py
"$BLENDER_BIN" --background --factory-startup --threads 1 --python-exit-code 1 \
  --python tests/blender_multiview_feedback_checks.py
```

旧的直接修正候选与候选排序入口已撤下；历史版本保留在 Git 提交记录中。其他微调辅助模块仅供兼容已有代码，当前运行方法以本节为准。

## 包络视角选择

独立的几何选视角工具：给定 BVB `.blend` 和题目，按目标联合包络采样相机，用视锥和遮挡射线选最少视角，再渲染。选视角阶段不读像素，不进入 200 题主协议。

可以从 `questions.jsonl` + `recipes.json` 自动生成题目目标和物体分组；也保留人工 `spec.json` + `object_groups.json` 入口。自动模式覆盖 6 类静态空间题，appearance order 和 route planning 会跳过并记录。

```bash
# 自动输入
"$BLENDER_BIN" --background --disable-autoexec /path/to/scene.blend --threads 2 \
  --python-exit-code 1 --python scripts/envelope_views.py -- \
  --questions data/questions.jsonl \
  --recipes data/recipes.json \
  --qa 1723 \
  --out runs/envelope_views/geometry_1723.json

# 或使用人工确认过的单场景输入
"$BLENDER_BIN" --background --disable-autoexec /path/to/scene.blend --threads 2 \
  --python-exit-code 1 --python scripts/envelope_views.py -- \
  --spec data/envelope_views/0d2ee665be/spec.json \
  --groups data/envelope_views/0d2ee665be/object_groups.json \
  --out runs/envelope_views/geometry.json
```

示例输入、输出字段和当前限制见 [`docs/envelope-view-selection.md`](docs/envelope-view-selection.md)。
