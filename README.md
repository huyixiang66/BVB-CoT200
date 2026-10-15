# BVB-CoT200：Blender 场景 → 八类 CoT → 单次答案

从已有 BVB Astra xhigh `.blend` 出发，运行固定的 200 道 VSI-Bench 问题（八类各 25 题、126 个场景）。**每题只请求一次，不做三次采样或投票。**

仓库包含代码、题目/评分标签、冻结的渲染配置与逐题 prompt；不包含 blend、原视频、渲染图片、API key 或历史实验目录。不是从视频重建场景，也不包含 debate 或场景微调。

## 环境

Python 3.11+、**Blender 5.2.2 LTS**（EEVEE 可运行）。Blender 使用自带 Python，不要用普通 Python 运行 `render_*.py`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export BLENDER_BIN='/path/to/blender'
export OPENAI_BASE_URL='https://your-provider.example/v1'
export OPENAI_MODEL='gpt-6-luna'
# 通过安全环境注入 OPENAI_API_KEY，不要把真实密钥写入仓库。
python run.py doctor
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`，环境变量使用 `$env:BLENDER_BIN='...'`。`.env.example` 只列出变量，程序不自动读取 `.env`。模型名和可用性以你的服务商为准，不自动回退模型。推荐 HTTPS；明文 HTTP 会暴露传输中的 key 和图片。

## 输入与协议

| 题型 | 渲染与 CoT |
| --- | --- |
| 绝对距离、相对距离、物体尺寸、房间面积、路线规划 | 最新 C：固定 BVB 自带相机、16 帧逐题高亮、新 CoT；不加俯视图、米制网格或数字答案标注 |
| 计数 | 页面 v2：俯视/斜视/分区近景、网格、高亮与跨帧实例去重 CoT |
| 相对方向 | 页面 v2：站立视角、俯视网格、近景与观察者方向 CoT |
| 出现顺序 | 页面 v2：BVB 相机时间线 24 帧、高亮与首次可见帧排序 CoT |

另外三类只保留 v2 的渲染和推理方法，**不保留投票**。q2583 无高亮匹配、实际只有 6 帧；已纠正其旧 prompt 的帧数说明，不补造物体。全量共 3,217 帧。

逐题文本在 [`data/prompts.json`](data/prompts.json)，配置在 [`data/recipes.json`](data/recipes.json)。固定的是 BVB **估计出的**相机，不是原视频真实位姿；名称匹配也不等于实例真值。

## 运行

建议先跑八题 smoke（每类一题），再使用独立目录跑全量：

```bash
python run.py download --smoke
python run.py render --smoke --run runs/smoke
python run.py validate --smoke --run runs/smoke
python run.py answer --smoke --run runs/smoke --workers 4
python run.py report --smoke --run runs/smoke

python run.py download
python run.py render --run runs/main
python run.py validate --run runs/main
python run.py answer --run runs/main --workers 4
python run.py report --run runs/main
```

只有 `answer` 调用模型，向配置的服务发送题面、CoT 和渲染图片；每题最多 8,000 completion tokens、300 秒超时、无自动重试。`--workers` 只控制不同问题的并发数，不增加采样次数。只测某题可加 `--only 720`，对应阶段需使用相同题目选择和独立 `--run`。

下载默认使用 Hugging Face，固定 revision 并校验文件大小/SHA-256。126 个 blend 约 2.42 GiB；国内可加 `--hf-endpoint https://hf-mirror.com`。已有文件按 `<scene_name>.blend` 命名，渲染时用 `--blends /path/to/blends`，无需重新下载，也无需原视频。

发布时本机镜像下载返回 403，真实远程下载未通过；网络不可用时直接使用已有 blend 目录，不要修改哈希或换未核对的资产。下载逻辑有 mock 测试，但不代表上游服务当前可访问。

Blender 串行运行，每进程两线程；不修改源 blend，不执行文件内嵌自动脚本。运行中请勿并发操作同一输出目录。成功结果校验后复用；配置/源码改变应使用新目录。

API 请求前写入 `requests.jsonl`，结束后写入 `answers.jsonl`。已记录成功或失败的题都不自动重发；如果中断发生在两者之间，程序会阻止续跑，避免对可能已计费的请求重复调用。先核查服务商记录与本地 PID，再处理该次运行；不要直接删除请求日志来“重试”。累计三次失败停止后续批次。

## 输出

| 文件 | 内容 |
| --- | --- |
| `renders/<qa_id>/prompt.txt`、`manifest.json` | 实际 prompt、帧路径/哈希、版本来源 |
| `logs/`、`render_config.json` | Blender 日志与冻结配置 |
| `answer_config.json`、`requests.jsonl` | 模型、输入参数与请求开始记录 |
| `answers.jsonl` | 单次答案、原始响应、实际模型名、tokens、耗时、状态；不保存 key |
| `per_question.csv`、`summary.json`、`REPORT.md` | 逐题判分与八类汇总 |

只有 `report` 读取评分标签。采用历史 BVB pilot 的语义准确率规则（多数数值题容差 20%），**不是 VSI-Bench 官方 MRA，也不是 dual-VQA**。缺失/失败题保留在分母中。

本发布组合尚未跑出新的 200 题成绩，不能沿用旧 87/200 或 91/200；完整来源和局限见 [NOTICE.md](NOTICE.md)。

## 不花 API 费用的测试

```bash
python -m unittest discover -s tests -v
python tests/smoke_blender.py --blender /path/to/blender
```

第一组验证数据完整性、调用协议、单次续跑和判分，API 为 mock。第二组从合成几何生成 blend，实际测试 C 和三类旧渲染路径，仅检查元数据与哈希，不查看图片。真实 200 题答题需自行执行上面的 `answer`。

主入口 `run.py`，调度/答题/报告在 `pipeline.py`，两套渲染器为 `render_fixed.py` 和 `render_legacy.py`。第三方许可见 `licenses/`，运行数据被 `.gitignore` 排除。
