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
