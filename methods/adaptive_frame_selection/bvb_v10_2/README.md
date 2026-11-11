# BVB V10_2 — 全轨迹和软空间锚点的联合重构

基于同仓库 V10 全流程，包含问题解析、2fps检测、LingBot位姿、光流、逐题选峰、DINOv2/几何聚类、Astra审核、联合重构、九视角选一及单图QA。原 V10 包与锁文件保持不变。本分支只包含方法代码、依赖、离线测试和说明，不提供实验结果、视频、图片、模型、生成场景或授权凭据。

## 本次改变

- LingBot关键帧间隔可配置。默认 `auto = max(2, ceil(N/320))`：475帧选2，868帧选3。320是经验性参考，不是容量保证；初始化和滑窗缓存另计。所有2fps候选帧仍输出位姿，关键帧间隔不等于抽帧率。
- 完整相机轨迹与选中帧位置/朝向绘成空空间俯视图、斜视图，和原帧一起交给Astra。
- 从检测框内的LingBot稀疏可见表面建立物体软锚点。复用 `bvb_v6.object_clustering.roi_surface` 的投影、前表面和空间分量筛选；至少12点才构成有效观测。逐帧中位数与10–90%分位范围按组汇总，中心用质量加权的逐坐标中位数。至少2个有效观测帧且中位启发式质量>=0.3标为medium，否则low；缺点时unknown。它不是物体分割、完整包围盒、统计置信区间或真实中心。
- Astra可依据原图调整或拒绝锚点；轨迹不是墙边界。不做强制局部场景配准、不钳制物体坐标、不自动迭代几何。统一相对尺度不是米制标尺。
- 默认仍是Astra high、subscription_direct、联合第0轮。目标物体保留结构；其他物体仅直接支撑/接触目标者用白块；保留所有有输入证据的墙、门洞边界、地板，不补屋顶或未见房间。
- 报告展示V10统一方法名；提供参考图、聚类帧、重构与QA。展示用GT不进入模型；失败或无有效答案计0分。

## 安装与完整流程

```bash
cd methods/adaptive_frame_selection
pip install -r requirements-v10_2.txt
# GPU环境、LingBot源码/权重、Blender隔离和OAuth授权见 bvb_v10/README.md
python -m bvb_v10_2 verify
python -m bvb_v10_2.workflow entities --questions /data/questions.json --out /runs/demo/entities
python -m bvb_v10_2.workflow features --video /data/video.mp4 \
  --questions /runs/demo/entities/questions.json --out /runs/demo/features \
  --lingbot-root /path/to/INSTALL --keyframe-interval auto
python -m bvb_v10_2.workflow run --questions /runs/demo/entities/questions.json \
  --features /runs/demo/features --out /runs/demo/v10_2 --device cuda \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
python -m bvb_v10_2.workflow report --out /runs/demo/v10_2 --reference /data/display-only-gt.json
```

`features --stage frames|dino|poses|flow`支持拆分GPU步骤，`--keyframe-interval 2`可固定间隔。改参数必须新目录，避免把旧缓存误认为新计算。报告位于运行目录的 `report/index.html`，每题结束刷新。输入题目与展示用GT格式沿用[V10说明](../bvb_v10/README.md)。现有QA适配器针对静态空间选择题，不声称支持任意题型。

## 已有审核/抽帧产物重跑

```bash
python -m bvb_v10_2 prepare --source /runs/old/prepared/QUESTION_ID \
  --poses /runs/new/features/poses --replace-poses --out /runs/new/question
python -m bvb_v10_2 run --out /runs/new/question --blender /Applications/Blender.app/Contents/MacOS/Blender
```

`--replace-poses`不重新抽帧或审核：保留图片和分组，逐帧核对source index、时间戳与SHA256，再替换精确相机和压缩VLM相机数据。未设置该开关时必须原位姿完全相符。报告保存新旧来源哈希、空间参考哈希及验证记录。`prepare`必须新目录，`run`复用已完成阶段；中断的API调用不会无条件自动重试。需要执行错误修复时显式 `infer --attempt attempt-02`，反馈只包含执行/schema错误，不使用GT纠错。

可选空场景：`blender -b --python bvb_v10_2/empty_reference_blend.py -- /runs/new/question`。它只画轨迹、选中相机与线框锚点，不修改重构网格。原始BVB全场景可用于独立人工对比，不能加入QA/重构输入。

## 限制与验证

运行环境沿用V10：Python>=3.11、2fps标准化640×480视频、GPU提特征、Mac `sandbox-exec`隔离Blender。Linux构建需要等价隔离适配，不能直接禁用沙箱。订阅直连需独立OAuth授权和对应模型访问权，不自动回退付费API。相机漂移、错误检测、背景点和遮挡仍会造成锚点误差；通过结构验证不代表几何准确。

```bash
python -m unittest bvb_v10_2.test_method
python -m unittest bvb_v10.test_portable
python scripts/test_bvb_v10.py
python -m bvb_v10_2 verify
```

发布检查使用离线测试与已有图片/位姿的prepare验证，不额外调用模型或重新跑GPU推理。新增源码单独锁定，并校验原V10依赖锁；源码变化应重新审查和冻结。
