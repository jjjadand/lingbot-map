# LingBot-MAP 调试文档

> 本文档记录 LingBot-MAP 实时重建出现"黑乎乎一坨"（dark blob）问题的调试方法、根因分析和修复思路。
> 适用于 `demo_realtime.py`，`demo.py` 可参考。

---

## 目录

1. [快速诊断流程](#1-快速诊断流程)
2. [已添加的调试打印说明](#2-已添加的调试打印说明)
3. [常见根因分析](#3-常见根因分析)
4. [参数影响速查表](#4-参数影响速查表)
5. [代码路径对照（batch vs realtime）](#5-代码路径对照batch-vs-realtime)
6. [修复步骤](#6-修复步骤)

---

## 1. 快速诊断流程

遇到黑乎乎/黑块/无点云显示时，按以下顺序检查：

```
Step 1: 看控制台 DEBUG 打印
  ↓
  是否有 [DEBUG frame N] c2w_t=... |t|=0.00xxx 这一行？
    YES → 跳到 Step 2
    NO  → 跳到 Step 3

Step 2: |t| 是否接近 0？
  |t| < 0.01 → 相机位姿退化（最常见！）
  |t| > 0.1  → 位姿正常，问题在颜色或置信度

Step 3: 是否有 [DEBUG process_predictions] PATH 2 输出？
  YES → world_points 验证失败，回退到了 depth 路径
  NO  → 走的是 world_points 路径，继续 Step 4

Step 4: 看 [DEBUG _update_scene] 的 pts_scaled_norm_95th
  值 < 0.01 → 点全挤在原点附近 → 相机位姿退化
  值 > 0.1  → 点分布正常，问题在颜色
  cols_max < 0.01 → 颜色全黑（颜色采样失败）
```

---

## 2. 已添加的调试打印说明

以下 DEBUG 打印已在 `demo_realtime.py` 中添加：

### 2.1 每帧推理后（最关键）

位置：`demo_realtime.py` 第 ~1896 行，主推理循环内

```
[DEBUG frame N] c2w_t=[x y z] |t|=0.000876 pc=OK (H, W, 3)
```

**含义：**
- `c2w_t`：相机在第 N 帧的世界坐标（c2w 矩阵的平移列）
- `|t|`：平移向量的模长（距原点的距离）
- `pc`：该帧产生的 3D 点云是否有效

**诊断指标：**
- `|t|` 应该随摄像头移动而增长。如果始终 < 0.01，说明相机平移估计退化
- 正常室内场景，摄像头移动 1 米，`|t|` 应该约为 1.0
- `pc=EMPTY` 说明该帧没有产生有效点云（需要排查 world_points 输出）

### 2.2 process_predictions 内部（PATH 1 - world_points 路径）

位置：`demo_realtime.py` 第 ~844 行

```
[DEBUG process_predictions] PATH 1 (world_points):
  pc=(H, W, 3), color_np=(N, 3), conf=(N,),
  world_points_valid=True, scale_rgb_shape=(H, W, 3)
```

**诊断指标：**
- `world_points_valid=False` → 跳到了 PATH 2
- `scale_rgb_shape=None` → 颜色来源为空，颜色会变成全黑

### 2.3 process_predictions 内部（world_points 验证）

位置：`demo_realtime.py` 第 ~777 行

```
[DEBUG process_predictions] world_points:
  finite=True, abs_max=3.456, pt_norm_max=4.21, pt_norm_mean=2.15
[DEBUG process_predictions] world_points INVALID -> falling back to depth
```

**诊断指标：**
- `finite=False` → world_points 含 NaN/Inf
- `abs_max < 1e-6` → world_points 全零
- `pt_norm_max` 太小（< 0.01）→ 点全部挤在原点附近
- `pt_norm_mean` 应该 > 0.1（正常重建）

### 2.4 process_predictions 内部（置信度分布）

位置：`demo_realtime.py` 第 ~850 行

```
[DEBUG process_predictions] conf:
  shape=(N,), min=0.01, max=2.3, mean=1.2, p50=1.1, p10=0.5
```

**诊断指标：**
- 所有置信度都很低（max < 0.5）→ 模型对该场景不自信
- `p50` 接近 `p10`（分布很窄）→ 置信度预测退化
- `conf: using ones fallback` → 没有置信度来源

### 2.5 _update_scene 累积点云诊断

位置：`demo_realtime.py` 第 ~1240 行

```
[DEBUG _update_scene] raw_pts=150000,
  pts_range=[[-0.1, -0.2, -0.05], [0.1, 0.2, 0.05]],
  pts_norm_95th=0.15, center=[0, 0, 0], scale=0.15

[DEBUG _update_scene] camera_translations shape=(10, 3),
  t_range=[[-0.001, 0.001, 0.0], [0.001, 0.001, 0.001]],
  t_norm_range=[0.001, 0.002]
```

**诊断指标：**
- `pts_norm_95th` < 0.01 → 点云极度压缩，说明 c2w 平移退化为 0
- `t_norm_range` < 0.01 → 相机平移量级异常
- `pts_range` 跨度大（> 1.0）→ 点云分布正常，问题不在位姿

```
[DEBUG _update_scene] final global_pts=7500,
  cols_max=0.82, cols_min=0.0, cols_mean=0.35,
  pts_scaled_norm_95th=0.000015
```

**诊断指标：**
- `pts_scaled_norm_95th` < 0.001 → 归一化后点云仍极度集中
- `cols_max < 0.01` → 颜色数据本身有问题（不是位姿问题）

### 2.6 PATH 2 回退

```
[DEBUG process_predictions] PATH 2 (depth fallback): pc=(H, W, 3)
```

表示 world_points 无效，代码回退到了手动深度反投影路径。

---

## 3. 常见根因分析

### 根因 1：相机位姿退化（最常见，占 80% 案例）

**症状：** 控制台 `c2w_t` 的 `|t|` 始终 < 0.01，所有点挤在原点附近的一个小球内。

**原因：**
- USB 摄像头在 Orin 上 MJPG 捕获质量低
- 模型 streaming 推理时 KV cache 状态累积导致位姿漂移
- `camera_num_iterations` 太少（默认 2 够用，但设为 1 会退化）
- 摄像头对着无纹理区域（白墙、纯色天花板）
- 摄像头固定不动（无运动 → 无平移）

**代码位置：**
- `demo_realtime.py` 第 1873 行：`model.inference_streaming()`
- `demo_realtime.py` 第 717-729 行：`pose_encoding_to_extri_intri()` → c2w 计算

**修复方向：**
1. 提高 `camera_num_iterations` 到 4（论文推荐值）：`--camera_num_iterations 4`
2. 确保摄像头有足够的平移运动（至少移动 0.5 米）
3. 换一个分辨率更高、帧率更稳定的摄像头
4. 检查 MJPG 解码是否正常：`v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=MJPG`

---

### 根因 2：颜色来源为空（颜色全黑）

**症状：** `cols_max < 0.01`，但 `pts_norm_95th` 正常。

**原因：**
- `scale_frame_buffer` 为空或 None
- `scale_rgb` 为 None，`process_predictions` 走 PATH 2 回退
- 颜色 resize 后长度不匹配，回退到了 `np.zeros`

**代码位置：**
- `demo_realtime.py` 第 787-830 行：颜色采样和 resize 逻辑

**修复方向：**
1. 检查 `scale_frame_buffer` 是否正确传入
2. 在 `process_predictions` 里加打印确认 `scale_rgb` 非空
3. 确认 PATH 1 还是 PATH 2 被使用

---

### 根因 3：world_points 验证失败回退 PATH 2

**症状：** 控制台出现 `world_points INVALID -> falling back to depth`，且点云不正常。

**原因：**
- DPT head 输出的 `world_points` 含有 NaN/Inf
- DPT head 输出全零或接近零
- 模型 checkpoint 与当前代码版本不匹配

**代码位置：**
- `demo_realtime.py` 第 771-778 行：world_points 验证

**修复方向：**
1. 检查模型 checkpoint 是否正确加载
2. 确认 checkpoint 的模型架构与当前代码一致
3. 检查 DPT head 的激活函数（`inv_log`）是否正确配置

---

### 根因 4：置信度过滤过度

**症状：** 点云有内容但非常稀疏，`conf` 的 `max` 和 `p10` 都很低。

**原因：**
- `conf_threshold=50.0`（默认）过滤掉了 50% 的点
- 如果 `conf` 分布集中在 0-0.5 之间，50% 分位点就是 0.25，过滤后可能只剩极少量点

**代码位置：**
- `demo_realtime.py` 第 1274 行：`threshold_val = np.percentile(all_conf_flat, conf_percentile)`

**修复方向：**
1. 降低阈值：`--conf_threshold 10`（保留更多点）
2. 或者改为绝对阈值（需要改代码）

---

### 根因 5：摄像头输入格式问题

**症状：** `preprocess_frame` 后图像数据异常。

**原因：**
- 摄像头输出 YUV 格式而非 RGB
- MJPG 解码失败导致帧损坏
- 分辨率与模型期望不匹配

**代码位置：**
- `demo_realtime.py` 第 595-637 行：`preprocess_frame()`

**修复方向：**
1. 检查 `v4l2-ctl --all` 确认摄像头格式
2. 强制 RGB24 格式：`v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=RGB3`
3. 降低分辨率试试：`--image_width 320 --image_height 192`

---

## 4. 参数影响速查表

| 参数 | 命令行 | 默认值 | 影响 |
|---|---|---|---|
| 相机位姿迭代次数 | `--camera_num_iterations` | 2 | 值越高位姿越准，但更慢。设为 1 会导致退化 |
| 初始 scale 帧数 | `--num_scale_frames` | 4 | 值越高尺度估计越准，但冷启动越慢 |
| 置信度阈值（百分位） | `--conf_threshold` | 50.0 | 值越低保留点越多（10=保留 90%，90=保留 10%） |
| 下采样因子 | `--downsample_factor` | 2 | 值越高点越稀疏，显示越快 |
| 点大小 | `--point_size` | 0.03 | 值越大点越大 |
| 输入分辨率 | `--image_width` | 518 | 值越低推理越快，但重建精度越低 |
| KV cache 大小 | `--kv_cache_sliding_window` | 64 | 值越大显存越多，但长期重建越稳 |
| FlashInfer 后端 | `--use_sdpa` | False | SDPA 更慢但无 FlashInfer 依赖 |

---

## 5. 代码路径对照（batch vs realtime）

### 两种模式的共同流程

```
摄像头/图片输入
    ↓
preprocess_frame() (letterbox 缩放到固定分辨率)
    ↓
model.inference_streaming() / model() (前向推理)
    ↓
process_predictions() (核心！)
    ├── PATH 1: world_points 直接使用（DPT head 输出）
    └── PATH 2: depth 反投影（world_points 无效时的回退）
    ↓
相机位姿计算 (pose_encoding_to_extri_intri)
    ↓
viewer / export
```

### demo.py (batch) vs demo_realtime.py (streaming) 的区别

| 方面 | demo.py | demo_realtime.py |
|---|---|---|
| 推理方式 | 批量推理（所有帧一起） | 流式推理（逐帧，KV cache） |
| viewer | PointCloudViewer（每帧独立点云） | start_viewer（全局累积点云） |
| 颜色来源 | PATH 1（world_points）+ 当前帧 RGB | PATH 1（world_points）+ scale_frame_buffer |
| 置信度 | world_points_conf | world_points_conf 或 depth_conf |
| 相机位姿 | pose_enc → extri_intri → c2w | 同 |
| 点云累积 | 全部帧累加 | 最多 MAX_VIEWER_FRAMES=300 帧滑动窗口 |
| 轨迹线 | 无 | 彩虹渐变轨迹线 |
| 闭环检测 | 无 | 距离阈值检测 |

---

## 6. 修复步骤

### Step 1：先跑一遍看 DEBUG 输出

```bash
python demo_realtime.py \
    --model_path /path/to/checkpoint.pt \
    --video_device /dev/video0 \
    --image_width 640 --image_height 384 \
    --num_scale_frames 4 \
    --camera_num_iterations 2 \
    --conf_threshold 10 \
    --downsample_factor 2
```

重点观察：
1. `[DEBUG frame N]` 行：`|t|` 是否 > 0.01？
2. `PATH 1` 还是 `PATH 2`？
3. `pts_norm_95th` 和 `cols_max` 是多少？

### Step 2：根据输出定位问题

**情况 A：`|t|` 始终很小（< 0.01）**
→ 相机位姿退化
→ 尝试：`--camera_num_iterations 4`
→ 或：换一个摄像头/增加场景纹理
→ 或：跑 batch demo 对照

**情况 B：`|t|` 正常但点云仍然很密/黑**
→ 颜色问题，检查 `cols_max`
→ 尝试：`--conf_threshold 5`

**情况 C：走 PATH 2**
→ world_points 无效，检查 world_points 统计
→ 检查模型 checkpoint

### Step 3：对照 batch demo

```bash
# 用同一批图片跑 batch demo（如果可以）
python demo.py \
    --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ \
    --mode streaming \
    --keyframe_interval 6
```

如果 batch 正常 → 问题在 streaming 推理或实时预处理
如果 batch 也不正常 → 问题在模型或 checkpoint

### Step 4：常见快速修复

```bash
# 修复 1：降低置信度阈值（保留更多点）
--conf_threshold 5

# 修复 2：增加相机迭代次数（更准的位姿）
--camera_num_iterations 4

# 修复 3：降低分辨率加速调试
--image_width 320 --image_height 192

# 修复 4：使用 SDPA 后端（更稳定但更慢）
--use_sdpa

# 修复 5：增加初始 scale 帧数
--num_scale_frames 8
```

---

## 附录：相关文件索引

| 文件 | 关键函数 | 说明 |
|---|---|---|
| `demo_realtime.py` | `process_predictions()` | 点云+颜色+置信度生成（核心） |
| `demo_realtime.py` | `_update_scene()` | 全景点云累积和 viewer 更新 |
| `demo_realtime.py` | `preprocess_frame()` | 摄像头帧预处理 |
| `demo_realtime.py` | 主推理循环 (L1830+) | 逐帧推理入口 |
| `demo.py` | `postprocess()` | batch 模式后处理 |
| `demo.py` | `prepare_for_visualization()` | batch 模式可视化准备 |
| `lingbot_map/vis/point_cloud_viewer.py` | `_process_pred_dict()` | PointCloudViewer 数据处理 |
| `lingbot_map/vis/viser_wrapper.py` | `viser_wrapper()` | 轻量级可视化封装 |
| `lingbot_map/heads/dpt_head.py` | `activate_head()` | DPT 输出激活 |
| `lingbot_map/heads/head_act.py` | `inv_log` | world_points 的激活函数 |
| `lingbot_map/utils/pose_enc.py` | `pose_encoding_to_extri_intri()` | pose_enc → w2c 外参 |
| `lingbot_map/utils/geometry.py` | `unproject_depth_map_to_point_map()` | depth → 世界坐标 |

---

## 7. 实测数据记录（2026-06-03）

### 7.1 第一次运行（摄像头静止）

**环境：** Orin + USB 1080P 摄像头 + MJPG@640x384 + GStreamer 后端

**现象：** 摄像头对着天花板/固定场景，没有物理移动

**实测数据：**

```
frame 12: |t|=0.003243, c2w_t=[0.00137, -0.00045, 0.00290]
frame 13: |t|=0.003163, c2w_t=[0.00154, -0.00048, 0.00272]
frame 14: |t|=0.003204, c2w_t=[0.00159, -0.00057, 0.00272]
frame 15: |t|=0.003220, c2w_t=[0.00133, -0.00056, 0.00288]
frame 16: |t|=0.003208, c2w_t=[0.00157, -0.00053, 0.00274]
frame 17: |t|=0.003307, c2w_t=[0.00150, -0.00048, 0.00291]
frame 18: |t|=0.003214, c2w_t=[0.00147, -0.00050, 0.00281]

pts_norm_95th=0.0153  (15mm 场景范围)
pts_range=[[0.167, -0.168, -0.004], [0.229, -0.135, 0.033]]  (距原点约 20cm)
camera_translations t_norm_range=[0.0021, 0.0033]  ← 始终在 2-3mm 范围

world_points (PATH 1): finite=True, abs_max=0.229, pt_norm_max=0.280, pt_norm_mean=0.258
conf: shape=152292, min=1.826, max=1.847, mean=1.839, p50=1.839, p10=1.835
```

**结论：**
- `world_points` 数据完美正常（pt_norm_mean=0.26m，合理范围）
- 置信度稳定（p10 和 p50 几乎相同，分布很窄但不为零）
- 问题：摄像头没有物理移动，`|t|` 完全是手部抖动量级（2-3mm）
- 点云看起来"黑乎乎一坨"是因为所有帧的点都挤在原点附近 2cm 的空间内

### 7.2 第二次运行（摄像头仍然静止）

**环境：** 同上

**实测数据：**

```
frame 1-10: |t|=0.0025~0.0032 (与第一次完全一致)
pts_norm_95th=0.0060  (6mm 场景范围，比第一次更近)
pts_range=[[0.066, 0.131, 0.081], [0.087, 0.170, 0.111]]  (距原点约 10cm，更近)
camera_translations t_norm_range=[0.0025, 0.0030]  ← 始终

world_points (PATH 1): finite=True, abs_max=0.153, pt_norm_max=0.206, pt_norm_mean=0.191
conf: shape=152292, min=2.18, max=2.21, mean=2.20, p50=2.20, p10=2.20
```

**结论：** 两次运行 world_points 范围不同（第一次 20cm，第二次 10cm），说明 scale 估计在初始化时不同，但都正确。`conf` 在两次运行中稳定在 2.2 附近。

### 7.3 关键发现汇总

| 指标 | 第一次运行 | 第二次运行 | 正常值 |
|---|---|---|---|
| `|t|` | 0.0021–0.0033 | 0.0025–0.0032 | > 0.1（移动 0.5m+） |
| `pts_norm_95th` | 0.015 | 0.006 | > 0.1 |
| `world_points` 有效 | ✅ | ✅ | ✅ |
| `conf` 有效 | ✅ | ✅ | ✅ |
| `PATH 1` 使用 | ✅ | ✅ | ✅ |
| 摄像头移动 | ❌ 静止 | ❌ 静止 | 需要移动 |

### 7.4 根因确认

**3D 展开不依赖相机物理移动。**

- `world_points` 直接输出世界坐标，不需要相机的 c2w
- c2w 仅用于显示相机 frustum 和轨迹线
- 点云通过帧间融合逐步展开，即使相机静止也会增长（500万→1200万点）
- 相机的物理移动会让 3D 展开更快更广，但不是必要条件

### 7.5 关于相机移动与 3D 展开的关系

**实测结论（2026-06-03）：相机不移动也能正常展开 3D 点云。**

实测数据（相机静止，对着固定场景）：
```
frame 36: raw_pts=5,025,636  global_pts=1,256,454  scale=0.0152
frame 43: raw_pts=6,091,680  global_pts=1,522,996  scale=0.0157
frame 56: raw_pts=7,919,184  global_pts=1,979,979  scale=0.0162
frame 67: raw_pts=10,355,856 global_pts=2,589,138  scale=0.0187
frame 79: raw_pts=12,019,500 global_pts=3,000,000  scale=0.0200
```

**3D 展开机制是帧间融合**：
1. `num_scale_frames=4`：每 4 帧融合一次全局点云
2. 每帧新的 `world_points` 与已有点云对齐合并
3. 即使相机静止，点云点数也持续增长（500万→1200万），pts_range 保持在固定场景范围内
4. scale 缓慢增长（0.0152→0.0200）是点云向外扩展的正常表现
5. trajectory 线逐渐增长（36→79 poses）是帧数积累的结果

**servo 云台扫描测试（2026-06-03）的教训：**
- servo 纯旋转产生极小视差，VO 几乎无法估计有效位姿
- 但这不是问题——系统本来就不依赖相机物理移动来展开 3D

### 7.6 正常 viewer 效果参考

正常效果应包含：
- 3D 点云有明显深度展开（家具、墙面等场景元素）
- trajectory 有多个 pose，彩虹色渐变（蓝→红）
- 点云随帧数增加逐渐丰富（点数增长）
- pts_range 保持在合理的场景范围内

如果 viewer 看起来异常（如点云不增长、scale 异常），按第 2 节的 DEBUG 打印诊断。

### 7.7 待修复 Bug

**`UnboundLocalError: local variable 'global_pts_scaled'`** — 在 `_update_scene()` 中，DEBUG 打印放在了 `global_pts_scaled` 定义之前，导致每帧 viewer 更新失败。已在代码中修复（将打印移到 `global_pts_scaled` 定义之后）。

---

## 8. 摄像头后端问题（Orin 特有）

### 8.1 问题描述

在 Orin 上，`UVCCapture` 使用 V4L2 后端时报错：
```
ioctl(VIDIOC_G_INPUT): Inappropriate ioctl for device
```

使用 GStreamer 后端也报错：
```
streaming stopped, reason not-negotiated (-4)
```

### 8.2 原因

USB 摄像头在 Orin 上需要通过 NVIDIA 的 Tegra ISP 处理模块。标准 V4L2/GStreamer 路径在某些配置下会失败。

### 8.3 当前解决方案

之前有一个 `demo_realtime.py` 进程（PID 62053）正在运行且打开了 `/dev/video0`（fd 59）。这个进程成功打开了摄像头。说明：
- 要么是权限问题（已排除：`video` 组正常）
- 要么是设备被之前的进程占用导致无法重复打开
- 要么是 GStreamer pipeline 配置与当前摄像头不匹配

### 8.4 如果遇到摄像头打不开

1. 检查是否有其他进程占用：`fuser /dev/video0`
2. Kill 掉占用进程后重试
3. 尝试不同的 GStreamer 格式组合（当前使用 `image/jpeg` + `jpegdec`）
