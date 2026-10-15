# BVB-CoT200：Blender 场景 → 八类 CoT → 单次答案

从已有 **BVB Astra / Luna xhigh `.blend`** 出发，运行固定的 200 道 VSI-Bench 问题（八类各 25 题、126 个场景）。场景来源与 QA 模型独立选择，**每题只请求一次，不做三次采样或投票。**

仓库包含代码、题目/评分标签、冻结的渲染配置与逐题 prompt；不包含 blend、原视频、渲染图片、API key 或历史实验目录。不是从视频重建场景，也不包含 debate 或场景微调。

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
