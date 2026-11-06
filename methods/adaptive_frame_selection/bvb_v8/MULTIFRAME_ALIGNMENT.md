# 多帧联合校正（实验性，可选）

继承的独立工具，未验证与 V8 最终实例裁选的组合效果，不属于默认执行链。

输入是已经准备和重建好的 V6 逐题局部场景、每帧相机位姿、检测框和已有拼合结果。该步骤不调用模型、不读答案标签，不修改输入场景，生成独立的候选拼合和校正前后图。原 `build` 流程仍保留直接按共享坐标拼合的行为。

```bash
cd methods/adaptive_frame_selection
python -m bvb_v8.multiframe_alignment \
  --prepared /path/to/prepared_question \
  --merged /path/to/build_output/merged \
  --blender /path/to/blender \
  --out /path/to/new_alignment_output
```

`--prepared` 需要 `manifest.json`、各局部组的 `input-frames.json` 和 `pose-reference.json`；`--merged` 需要原 V6 build 的 `merge.json` 和 `merged-scene.blend`。输入相机必须是该构建使用的 Blender C2W、对应 640×480 图像的内参；并不支持任意坐标约定。可选 `--detections /path/to/detections.jsonl` 从缓存补充原输入中没有框的帧；调用者必须确认它来自**同一原视频与相同的帧编号**。不传时仅使用原输入帧中的检测框。

每组只有一个可识别目标时，程序从现有目标网格导出顶点，将其投回同组多帧，联合拟合有界的三维平移与水平旋转；截断在图像边界的检测框采用单侧约束。至少两帧参与拟合、另外一帧留出作诊断；初始投影无效的帧被排除。范围为横向各 ±0.75、竖向 ±0.35 场景单位，水平旋转 ±20°，这些是实验约束而非米制精度。场景形状、帧和相机位姿不重新估计，同一局部组的墙与上下文随组整体移动。

输出 `alignment.json`、`before-top.png`、`before-oblique.png`、`aligned-top.png`、`aligned-oblique.png`、`aligned.blend` 和 `aligned-merge.json`。每对图共享相机，取景覆盖前后场景。多个目标位于同一组且无法匹配检测框、目标缺失或有效帧不足时，记录 `skipped` / `insufficient` 并保留原位。拟合失败的组也不应用变换。无论投影误差是否下降，这些输出都是**人工核查候选**：留出帧可能与拟合帧相邻，检测框、位姿和目标形状也可能有误，不能把它当作已验证的几何改善。该步骤不做墙体配准、跨组去重或 QA。

数值检查：`python -m unittest bvb_v8.test_multiframe_alignment bvb_v8.test_question_policy bvb_v8.test_coverage`。
