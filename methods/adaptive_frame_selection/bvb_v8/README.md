# BVB V8：整题目标重构与整合时实例选择

版本 8.1.0。基于上游提交 `011ccf10acfe7facb99456ccd222f31d2c51fedf` 的 V6 方法，新增独立 V8 命名空间；原 V3–V6 不变。仅包含源码、测试、依赖和许可，不包含实验数据、媒体、权重或密钥。

## 方法

1. 输入公开题干、全部选项和相关物体列表。复用 0.5 主门槛选峰缓存，不开启跨类别波峰叠加；此包不会把低门槛输入缓存自动恢复为 0.5。
2. 沿用光流互补帧、外观/位姿聚类和干净原图语义审核。单实例题保留多个审核通过的视角组，实例选择延后到整合。
3. 每个局部重构收到整题物体列表以及当前主要目标。图中其他相关物体也可构建，不要求它已有独立波峰；不可见物体不得补造。
4. 仅重构题目目标、直接物理支撑目标的白块、墙和地板。context 只允许 support，不允许邻近参照、容纳、附着、障碍物或支撑链递归。支撑物若本身是目标，不重复生成白块。地板按图像和位姿确定局部范围/高度，用白色薄板表示，省略屋顶和天花板。
5. 全部局部 JSON 齐备后，resolve-instances 把去重的干净原图、候选几何和共享位姿交给 Astra。它判断实例成员、排除错误指代，并选择一份已有几何作为每个实例的代表。其他成员的帧与局部文件仍保留。
6. build 校验实例数、成员完整性和输入哈希，再过滤构建副本。移除重复目标及失去目标引用的白块，保留墙和地板。目标按自身类别着色，不按所属主目标着色。

## 保留策略

明确单指代的相对方向、绝对距离和尺寸题，默认每类别最终保留一个实例。计数和相对距离题允许多个。复数、同类多角色或未知题型不擅自压成一个，不能从 STI 数据集名称假定没有重复物体。公开题目元数据可用 instance_limits 指定每类正整数或 null；retention_policy 仍可覆盖题型策略。

有候选时至少保留一个；不确定时标记 provisional，不能冒充确认。完全没有候选则报告缺失，不补造。all_groups_compiled 只说明编译；target_coverage_complete、final_scene_complete、provisional_instances 分别记录目标覆盖、构建完整性和暂定实例，都不是几何或 QA 正确率。

## 运行

在 methods/adaptive_frame_selection 下执行。Blender 独立安装；检测 GPU 依赖见 requirements-gpu.txt。

```bash
pip install -r bvb_v8/requirements.txt
python -m bvb_v8 --help
# select / semantic_review / coverage 接口沿用原方法，改为 bvb_v8 命名空间。
python -m bvb_v8 prepare --selection /path/reviewed-selection.json --qid QUESTION_ID --cache /path/cache --poses /path/poses --out /path/prepared
python -m bvb_v8 submit --out /path/prepared --env-file /path/private.env --limit NUMBER_OF_GROUPS
# 仅生成计划，不调用 API：
python -m bvb_v8 resolve-instances --source /path/prepared
# 显式调用官方 Astra high 作实例判断：
python -m bvb_v8 resolve-instances --source /path/prepared --execute --env-file /path/private.env
python -m bvb_v8 build --source /path/prepared --out /path/build --blender /path/to/blender
python -m unittest discover -s bvb_v8 -t .
```

密钥仅从环境变量 LLM_API_KEY 或显式本地 env 文件读取。重构和整合默认不设置 API 输出 token 硬上限，保留提示词的简洁输出要求。整合新增一次付费调用，usage 写入 instance-resolution/request.json；已有账本不会自动重复提交，失败需检查后安排新运行目录。

必须重新 prepare 使用新提示词；旧单实例策略裁掉的视角需重新 coverage/export。旧结果不自动升级。决议校验绑定全部输入/图像/提示词；过期或违反数量限制的决议不能构建最终场景。该整合选择现有几何，不做平均、ICP 或联合优化，也不去重墙和地板，局部地板可能重叠。真实身份和物理支撑关系仍依赖模型判断。原多帧校正为独立实验工具，未验证与新实例裁选的组合效果。
