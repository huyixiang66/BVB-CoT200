# Room-size：V10_2 重建适配器

本版恢复学长 V10_2 的联合重建流程，通用化已存档81.82%实验中的原帧尺度估计、固定区域尺寸、俯视渲染和参照上沿QA。已删除上一版“家具组件JSON→新建模器”的执行路径。

没有题号或场景名分支，不输入GT/历史答案。50题新场景实验已结束，47题完成方法QA；81.82%仍只属于冻结七题结果，不是这份通用代码的新测分数。

## 完整流程

1. **选帧**：从已有LingBot缓存对应的原图中选择时间/相机观察方向互补的帧。以 `ceil(duration × 0.125)` 为软参考，新增收益不足时停，不设硬上限。复制原缓存JPEG并核对sampling SHA，不重新编码后硬套位姿。该选帧策略已用于50题实验；几何多样性不保证语义覆盖完整。
2. **原帧尺度估计**：Astra High看选中原帧，给出区域长宽、连通关系和区域内参照尺寸范围。证据不完整时给出最佳数值估计并记录不确定性；无有效数值时阻塞，不自动收费重试。估计表不含总面积答案。
3. **V10_2证据准备**：直接调用上游 `bvb_v6.reconstruction.shared_pose`、`compact_pose_input.build_transport`、`bvb_v10_2.reference.build`。沿用完整轨迹、选中相机、DINO框内软表面锚点及俯视/斜视参考图。适配器读取指定缓存；本轮50题在上游重新运行LingBot，DINO复用。不把轨迹当墙，不把点云当真实米制。
4. **V10_2联合重建**：直接使用上游 `bvb_v10.prompt.build`＋`bvb_v10_2.pipeline.EXTRA`，再追加从旧实验泛化出的固定尺寸说明。原帧与对应compact_camera/observations交错输入，末尾附软锚点和两张空间参考图。Astra输出完整Blender Python，自行构建可辨认的家具多组件；不再输出受新JSON形状表限制的家具。
5. **固定尺寸后处理**：沿用实验中 `enforce.py` 的职责：读取模型给出的region_layout/openings，按原帧估计尺寸创建墙地板，验收不重叠、连通和家具越界。**保留生成家具的原网格，不重建或单独缩放家具**。实际参照尺寸须在原估计范围内；测量整个instance的部件，而不是单个mesh。仍允许度量布局相对原相机旋转/平移，不宣称严格保留LingBot重投影。
6. **俯视图**：全屋正交俯视、1200×1000、6%余量，保留墙/地板，沿用旧渲染的材质与阴影设置。参照位置和完整尺寸自动整理成文字，不能可靠定位/看清则停止QA。
7. **QA**：只给这张俯视图＋一个明确参照的实际采用尺寸及“原帧估计、非实测”的来源；使用冻结的合理区间上沿提示。不传原帧、区域尺寸表、诊断面积或GT。

没有调用上游完整 `workflow.run`，因为其默认QA只接受选择题。本适配器复用其**重建方法和输入形式**，替换数值room_size的准备和QA入口。订阅传输复用项目已有客户端，Astra High固定；不静默换模型。

## 与旧实验的关系

- 上游版本：`methods/bvb-v10_2-full-pipeline-20261110`，核对的提交 `7568fce70c45`，基于V10提交 `5e2355e157f6`。运行时从配置指定的完整checkout直接加载，不复制另一份建模实现；`run.json`记录实际上游源码SHA。
- 固定区域尺寸后处理源自旧 `room-2697-hard-dimensions` 和 `room-five-hard-dimensions` 实验的enforce.py，去掉具体区域名称、尺寸、连接和题号分支。
- 参照尺寸提示、统一上沿提示沿用67.53%→81.82%实验的做法。
- 旧实验有人工整理的分块/参照和2701手工布局。本版把这部分交给模型结构化估计及通用检查，不声称自动化后数值与旧结果相同。
- 俯视参照自动定位和选帧是通用衔接逻辑，不属于已经测得81.82%的自动算法。几何验收证明遵守估计约束，不证明估计符合真实房间。

## 配置与运行

在Linux实验服务器运行。依赖：Python 3.10或以上（使用zip strict语义）；完整上游 `methods/adaptive_frame_selection` checkout及其Python依赖（numpy/scipy/matplotlib等）、ffprobe、Blender、bubblewrap。生成的Python在断网只读系统沙箱中运行，仅输出目录可写，两CPU线程、nice15；无不隔离的回退路径。订阅客户端按自己的说明安装依赖，不附带任何凭据。

在本目录执行，复制并修改 `examples/config.json`：

- `video`：原视频，仅用于时长及SHA记录。
- `v10_root`：上游 `methods/adaptive_frame_selection`目录，需同时含bvb_v6/bvb_v10/bvb_v10_2。
- `poses`：已有新LingBot缓存，含camera-poses.npz、sampling.json、sparse-geometry-corrected.npz，及可用的geometry-info.json。
- `frames_root`：sampling.json里的filename相对该目录解析；每张入选图SHA必须一致。视频与缓存的配对由配置提供，程序不会重新判断两段视频语义是否相同。
- `detections`：原缓存detections.jsonl，source_frame_index须与sampling一致，沿用其object标签与框。省略时没有DINO软锚点，不应称为完整证据等价运行；正常实验应提供此文件。
- `question`：原题文本，不传含GT的整个数据集记录。
- `out`：新的实验目录；其余见示例。

```sh
export PYTHONPATH="$PWD/src"
python -m room_size_pipeline plan --config config.json
python -m room_size_pipeline run --config config.json --until select
python -m room_size_pipeline run --config config.json \
  --subscription-client /path/to/approved-client \
  --profile YOUR_PRO_PROFILE --expected-email YOUR_ACCOUNT --pro-confirmed
```

Windows只做代码检查和plan，建议 `python -X utf8`；实际建模留在服务器。路径相对配置文件目录解析。支持 `--until select|estimate|reconstruct|render|qa`，可先审核Python再运行渲染。一次新完整流程正常为估计、重建、QA三次模型请求；收费请求前请检查输入及账号。

Pro身份签名验证、模型目录检查沿用客户端；套餐需操作者确认。现有订阅路线不暴露剩余额度，必须事先核对，参数不是额度证明。失败/中断保存状态及未知用量，不自动重试。成功阶段可恢复复用；代码/缓存/视频变化必须新目录。共享服务器凭据按项目规则在实验后清理。

## 产物

`selected.json`与frames保存抽帧及位姿；estimate保存原帧尺寸估计；v10_input保存上游输入/空间参考图/实际prompt；reconstruct保存请求、响应和generated.py；render保存joint-original.blend（后处理前）、scene.blend、top.png、geometry.json和reference.json；qa保存实际单图请求、回答和usage。

诊断面积只在geometry.json中，不进入最终QA。没有从顶点面积直接计算最终答案。

## 验证

```sh
python -m unittest discover -s tests -p test_room_size_pipeline.py -v
```

默认入口已整合执行/几何修复，无需切换repair模块。22项回归中21通过、1因Windows符号链接权限跳过：软预算、转身观察、固定区域/连接/参照范围、失败不重试、上游版本变化阻断恢复、Python重建输出和QA输入隔离。另直接加载真实上游代码，在合成输入上生成了2张V10_2参考图并核对原prompt/EXTRA；服务器合成场景通过生成Python→保存blend→固定结构→俯视渲染路径，8顶点床垫与114顶点枕头原组件均保留。零模型调用、没有真实题目重跑。后续真实实验及局限见 HANDOFF.md。

## 默认修复已接入

直接使用上面的run命令即可。区域开口、实例测量范围/姿态、俯视可用轴及Blender5引擎名称兼容均走默认入口。repair_*.py仅作向后兼容；细节和元数据边界见HANDOFF.md。新代码请使用新输出目录，历史源码哈希保护不绕过。
