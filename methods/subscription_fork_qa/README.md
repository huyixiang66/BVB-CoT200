# 本地订阅代理：给实验代码调用的原型

本模块提供一个运行在自己电脑上的服务，把实验代码的 OpenAI 风格请求转换成 Codex app-server 请求，使用本机已登录账号的订阅额度。服务直接调用本机 Codex，不经过 CLIProxyAPI 作者的中转；Codex 本身仍需要连接 OpenAI。它是协议兼容的**有限子集**，不是完整的 CLIProxyAPI，也不保证所有模型、参数和实验能直接工作。

**当前验证状态：尚不适合直接跑全量。** 协议、OpenAI SDK 和独立 fork 已用本地假服务测试；本地服务的一次真实 Plus 图片短测成功返回 JSON，但没有识别图片，返回了 `unknown`。随后已把图片转换成 Codex 原生 `localImage` 格式；该修改通过离线测试，**尚未再次真实验证**。先用真实图片做小额 smoke test，确认视觉、目标模型和任务输出，再逐步扩大实验。另一条独立 fork 脚本曾做过真实图片和双分支试验；不能把它的结果当成本代理新版本已经验收。

## 启动本地服务

需要 Python 3.10+、Codex CLI 0.167.0+，以及使用各自账号完成 `codex login`。执行 `codex login status` 检查登录。示例为 Windows PowerShell，在仓库根目录执行：

```powershell
$env:BVB_PROXY_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:LLM_API_KEY = $env:BVB_PROXY_API_KEY
$env:LLM_API_BASE = 'http://127.0.0.1:18317/v1'
python methods/subscription_fork_qa/subscription_proxy.py --max-model-turns 10
```

服务只监听本机，密钥是客户端访问本地服务的随机口令，**不是 OpenAI API key，也不是账号登录凭证**。另开终端运行客户端时，需要在那个终端设置相同的 `LLM_API_KEY` 和 `LLM_API_BASE`。不要把口令、登录文件、session ID 写入仓库或共享日志。`--max-model-turns` 默认 10，父对话、每次回答和失败尝试都计入，服务重启后重新计数。退出服务使用 Ctrl+C。

现有 V6 的重建 submit、coverage 和 semantic review 会读取 `LLM_API_BASE`；未设置时仍走原有 OpenAI 地址。模型名需指定为账号和 Codex 实际支持的模型；服务的 `/v1/models` 列表仅是本地配置，**不证明账号拥有该模型**。V6 原代码发送 `max_tokens`，而 Codex app-server 不能保证输出 token 硬上限，因此默认拒绝这样的请求；如理解并接受可能超出原预算，才启动时显式加 `--allow-uncapped-output`。请求里的 token 上限此时不会被执行，重建本身也需要另外验证；建议先针对一组小输入使用 `--limit 1`。

## 普通单次请求

安装 OpenAI Python SDK 后，现有代码可将 `base_url` 和 `api_key` 指向上述本地服务：

```python
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ['LLM_API_KEY'],
                base_url=os.environ['LLM_API_BASE'], max_retries=0)
response = client.chat.completions.create(
    model='gpt-6.1-sol',
    messages=[{'role': 'user', 'content': 'Reply OK.'}],
)
print(response.choices[0].message.content)
```

这是真实模型调用，会消耗订阅额度。可用 OpenAI `image_url` 内容块传入 `data:image/...;base64,...` 格式的图片，支持 PNG/JPEG/WEBP/GIF；不支持远程图片 URL 或原始视频文件。客户端需先自行抽帧。每次普通调用是独立对话。

## 共用帧并为每题 fork

以下使用额外的 `requests` 包；父对话发送图片和公共上下文一次，每道题通过独立子分支提问：

```python
import base64
import os
from pathlib import Path
import requests
from openai import OpenAI

base = os.environ['LLM_API_BASE'].rstrip('/')
key = os.environ['LLM_API_KEY']
headers = {'Authorization': 'Bearer ' + key}
image = base64.b64encode(Path('frame.png').read_bytes()).decode('ascii')
parent = requests.post(base + '/fork/sessions', headers=headers, timeout=180, json={
    'model': 'gpt-6.1-sol',
    'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Use these shared frames to answer later questions.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}},
    ]}],
})
parent.raise_for_status()
session = parent.json()
try:
    fork_client = OpenAI(api_key=key, base_url=base + '/fork/sessions/' + session['id'],
                         max_retries=0, timeout=180)
    for question in ['Question 1?', 'Question 2?']:
        result = fork_client.chat.completions.create(
            model='gpt-6.1-sol', messages=[{'role': 'user', 'content': question}])
        print(result.choices[0].message.content, result.usage)
finally:
    requests.delete(base + '/fork/sessions/' + session['id'], headers=headers, timeout=30).raise_for_status()
```

父请求只放公共图片和背景，不放任何题目的 GT 或答案；每个子请求只放本题。`parent_usage` 和各子请求 `usage` 都要纳入统计，失败时未知消耗不能记为零。删除 session 清理服务内的临时图片和引用，但不保证删除 Codex 持久化的父会话记录。session ID 具有访问能力，避免记录完整 URL。

**fork 不等于服务端图片只上传一次或一定省额度。** 本地模拟发现分支可能再次上传相同图片；之前真实 fork 出现过缓存命中，但命中的内容、比例和订阅额度节省都未证实，不能据此宣传节省比例。请求中还包含 Codex 指令等上下文，曾观测到约 2 万输入 token。不能保证缓存命中，也不会在 ChatGPT 网页中创建可见对话框。

## 接口边界

- `POST /v1/chat/completions`、`POST /v1/fork/sessions`、`POST /v1/fork/sessions/{id}/chat/completions`、`DELETE /v1/fork/sessions/{id}` 和 `GET /v1/models`；本地服务串行执行，无自动重试。
- 支持一个 user 消息及其之前的文本 system/developer 消息；图片使用内联 data URL。支持文本及 `json_schema` 输出格式；schema 传给 Codex，代理仅检查返回内容是合法 JSON，不独立保证完全符合 schema。
- 不支持流式输出、工具调用、多轮消息数组、远程图片 URL 等；不支持的参数会拒绝，不会假装生效。`max_tokens` 和 `max_completion_tokens` 默认拒绝；开启 `--allow-uncapped-output` 后才接受但不执行限制。返回 `model` 是请求值，不是服务端模型的独立验证。
- 本地计数和返回 token usage 只用于观察，不能推算 Plus/Pro 实际额度、价格或省钱比例；可能出现缺失或失败后未知消耗。请遵守所用账号的适用条款，不共享账号凭证。

## 不启动真实模型的测试

```bash
python -m unittest discover -s methods/subscription_fork_qa/tests -v
```

测试使用假 app-server 与模拟图片，不消耗订阅。OpenAI SDK 相关测试在安装 SDK 时运行。下面的原始批量脚本可作为独立入口使用。

---

# Independent question forks with a ChatGPT subscription

## 中文说明

这个模块用于把**已经选好的同一组图片和多道问题**接入 Codex：先把图片送入父对话，再为每道题创建互不影响的子对话，保存答案与缓存用量。它不负责视频抽帧、场景重建、选视角或评分，也不会直接输入视频文件。输入文件格式见 [example.json](example.json)，每题的选项可写在 `question` 中。默认最多处理两题；先加 `--validate-only` 检查，不会调用模型。

目前已实测一张图片、两个独立分支和共同上下文缓存命中；不代表视频图片已经只上传一次，也未证实 Plus 额度的节省比例。假上游测试显示，HTTP 模式下同一图片可能随每个 fork 重传。后续请先小规模验证，再决定是否批量运行。

This prototype starts a Codex app-server signed in with the user's own ChatGPT account. It sends a shared set of images to a parent conversation, then creates an independent fork for each question. Answers, status, elapsed time and reported token usage are written to a local JSON file. Ground truth is not read or included.

This is a **QA adapter**, not the full video-reconstruction pipeline: detection, frame selection, clustering, Blender reconstruction and view rendering must happen upstream. Input images can be selected source-video frames or rendered scene views. A manifest represents one shared image set and the questions to answer from it; questions needing different image sets must use separate manifests.

## Prerequisites

- Python 3.10 or later; only the standard library is used.
- An officially installed Codex CLI signed in to the intended ChatGPT account. Check with `codex --version` and `codex login status`. Codex 0.167.0 was used for the successful pilot; the runner rejects older versions before inference and the experimental parameters may change in newer versions. No API key, OAuth token, or network proxy is supplied by this project.
- The intended account and available usage must be checked by the operator. This consumes ChatGPT/Codex plan usage, subject to its limits and applicable terms. Do not share account credentials, bypass limits, or commit credentials.

## Prepare input

Create a JSON file like [example.json](example.json). Image paths are relative to that JSON file; images must already exist. Include one to six images, plus a unique `id` and full question text (including choices when applicable) per question. Do not put reference answers or evaluation labels in the question text. The runner reads only `images`, `id`, and `question` and does not calculate benchmark scores.

```bash
python methods/subscription_fork_qa/run_fork_qa.py path/to/input.json --output path/to/results.json --validate-only
python methods/subscription_fork_qa/run_fork_qa.py path/to/input.json --output path/to/results.json --limit 2
```

Validation starts no Codex process and makes no inference calls. The execution default is **two questions**, plus one parent turn; set `--limit` explicitly to change it. Use a new output path: existing results are never intentionally overwritten. `--codex` selects an executable or a Node-installed Codex `.js` entry point; `--model` selects an available model ID. Runtime network settings come from the operator's environment; no machine-specific proxy or personal path is embedded.

Each question is forked independently from the same parent via `experimentalPredictionMode`, `ephemeral` and `excludeTurns`. This is an experimental Codex protocol, not a documented guarantee of cache sharing. It creates Codex sessions and does **not** guarantee that ordinary ChatGPT web conversations appear. Parent session history may be stored by Codex; the child forks are ephemeral.

## What was actually validated

- With one synthetic image and two independent forks on Plus, answers were correct and independent, and each fork reported substantial cached input. A normal fork and an experimental fork subsequently reported the same cached input on a single comparison.
- A local fake HTTP upstream showed the image being sent to that upstream again for each fork. **Do not claim the image is uploaded to the model only once.** Actual ChatGPT WebSocket traffic was not independently measured.
- Cache hits are not guaranteed; aggregate `cachedInputTokens` does not prove how much of an image was cached. The real pilot carried about 20,000 input tokens per turn including Codex context. It has not established Plus quota savings, actual cost, stability at scale, ten-question correctness or scientific QA accuracy.
- The JSON records `usage: null` when usage is unavailable. A failure stops the run and is recorded without automatic retry; any unreported consumption remains unknown. `model_requested` is the selected ID, not an independently verified server model receipt. Assess total workflow costs separately, including construction and failures.

The code does not include the experimental inputs, videos, `.blend` files, images, credentials or model outputs. Do not publish real datasets or private local paths without checking their license and privacy requirements.
