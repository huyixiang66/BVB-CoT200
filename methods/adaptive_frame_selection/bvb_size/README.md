# Object-size pipeline：交接与发布准备

本模块是 object-size 实验的分阶段实现，依赖同仓库的 V10 前端与 `methods/room_size_pipeline/src/room_size_pipeline` 调用后端。

## 实验流程

1. **前端**：给出视频和问题，复用或运行上游 V10 的检测、光流、峰值筛选和聚类；LingBot 提供与原帧绑定的相机位姿。候选检测不保证是真目标，缓存应核对场景、帧号、时间及哈希。上游源码、模型和特征不在本快照内。
2. **目标和参照审核**：从候选组选出真实目标；按 `target_reference_policy.json` 和 `priors.json` 决定是否找同帧参照。类别规则来自 20 题开发实验，未知类别目前回退为仅目标；模型确认和边界判断并不保证真实正确。审核仅目标组的代表帧，必要时应补充其他候选组。`2318`、`2383` 用了其他题同场景缓存帧重新审核。
3. **联合重建**：将所选组所有原帧、逐帧位姿和目标/参照角色送入模型，生成 Blender 代码。使用 V10 的模板和数据约定做适配，不能称作完整未修改的 V10 端到端复现。构建应在 Linux bubblewrap 隔离环境进行，禁止直接无沙箱执行生成代码。
4. **选视角和人工检查**：有参照时在场景中优先尝试同图，不满足约束时尝试同焦距、朝向和距离的两张图；仅目标时使用独立目标渲染，优先选择减少水平长边透视压缩的方向。若两种联合视角均不合格且射线确认被结构墙挡住，暂时隐藏实际击中的墙并重新求解；通过时记录被隐藏的墙，未通过则保留原始诊断结果。不会因太小或被非墙物体遮挡而自动去墙，也不会移除地板或所有上下文。渲染与原始场景均不会改变物体缩放。几何通过仍要查看图像：遮挡、侧视、物体身份和比例都可能出错。`unresolved` 不能作为已通过视角的 QA 输入。
5. **QA**：检查通过后输入建模图以及该组全部原始重建帧，再附原问题；本次使用 Astra High，每题保留输入、结果、失败和未知消耗。最后用 VSI MRA 数值评分。失败不可记成零分；失败的重试需独立目录，成功的调用不重复。

62 题的原帧每题 4 或 8 张，重建图通常 1–2 张。`2366` 的参考物视角没有通过，使用单目标视角及原帧的回退输入；其模型形状和目标范围有疑问。部分其他题的识别或重建比例也存在不确定性，需保留审核记录。

## 文件

- `methods/adaptive_frame_selection/bvb_size/`：证据导出、参照策略、审核及输入准备、生成模型调用、隔离构建、渲染、视觉审核、QA、MRA 和测试。
- `methods/room_size_pipeline/src/room_size_pipeline/`：同仓库共享的后端与调用辅助模块，object-size 不使用 room-size 的建模方法。
- 视角修复已接入默认流程：单目标方向改进，以及按射线检测结果选择性隐藏结构墙的重试。默认修复与历史隐藏墙和 context 的个别实验变体不是相同实现。
- `vsi-size-62-results.json`：62 题原始预测、GT、MRA 和 0.125 FPS 对照；不含原帧、模型响应、个人路径及凭据。可以按需决定是否发布数据。

## 使用边界

代码中的阶段接口为 `python -m bvb_size.batch_workflow --help`，另有 `handoff`、`export_evidence` 和 `visual_review` CLI。入口从上游已准备的 evidence 和模型模板开始；前端特征提取、证据生成、跨机器传输、补检、视角不合格后的重渲染及备用目标搜寻仍需调用方串联。订阅后端需要外部客户端和用户自行授权，账户信息和凭据不随包提供；也可以通过实现后端接口接自己的模型。

默认构建与视角阶段要求 Linux、bubblewrap 和 Blender；先在目标机器验证隔离和资源限制。`linux_build.py --probe` 的个人路径断言已删除，只做写入探针，不能作为完整隔离验证。

## 实验结果及口径

VSI 七场景 62 题，本轮方法 MRA **77.42%**，0.125 FPS 原帧 baseline **75.22%**。Baseline 汇合历史结果和补跑结果，部分历史模型/提示词配置未完整留存，因此这不是完全同配置的严格对照；参照规则也来自小规模开发数据，不能据此推断跨数据集泛化。QA 输入帧数更少不等于完整流程总 token/成本更低，审核和重建调用也需计入。

## 基础检查

```sh
cd methods/adaptive_frame_selection
export PYTHONPATH="../room_size_pipeline/src:$PYTHONPATH"
python -m unittest bvb_size.test_policy bvb_size.test_handoff bvb_size.test_inference bvb_size.test_workflow bvb_size.test_view_repair
python -m bvb_size.batch_workflow --help
```

真实视频、LingBot/DINO 权重和缓存、原始输入帧、Blender 场景、账号、密钥及本机/服务器专用脚本不包含在这个待发布快照中。

视角修复已通过离线单元测试，但此次合入前尚未对新自动重试在真实 Blender 场景重新做端到端渲染验证。
