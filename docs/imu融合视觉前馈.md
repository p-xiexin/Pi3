# IMU 融合视觉前馈

目标是在 MASt3R-Fusion 的视觉惯性框架中使用 Pi3X 的多视图点图与位姿预测。改造分为三个部分：抽象前馈模型接口，用简单光流选择关键帧，再通过延迟窗口优化融合视觉与 IMU。

实现位于 [pi3x-delayed-matching 分支](https://github.com/p-xiexin/MASt3R-Fusion/tree/pi3x-delayed-matching)。方法参考 [MASt3R-Fusion 论文](https://arxiv.org/abs/2509.20757)与 [Pi3 / Pi3X 代码](https://github.com/yyfz/Pi3)。

## 三个核心改造方案

### 1. 前馈模型的抽象

原始 MASt3R-Fusion 的跟踪和因子构建直接调用 MASt3R 推理与匹配函数。改造后，`FeedForwardFrontend` 定义统一接口，`load_frontend_model()` 根据配置加载 `MASt3RAdapter` 或 `PI3Adapter`。模型差异在 adapter 内转换为点图、置信度和对应索引，跟踪与后端通过这些输出构建约束。

核心方法及两条模型路径如下：

| 方法 | MASt3R 内部处理 | Pi3X 内部处理 |
| --- | --- | --- |
| `get_feature_spec()` | 返回共享特征布局，patch size 为 `16` | 返回 `feat_dim=1024, patch_size=14`，用于分配共享帧缓冲区 |
| `encode_frame(frame)` | 调用 `_encode_image()`，将编码特征与位置存入 `frame.feat/pos`；已有特征时复用 | 对图像归一化后调用 encoder，保存 patch tokens，并生成 patch 网格位置 |
| `infer_single(frame)` | 将同一帧的特征作为两路输入，经 `_decoder()` 和 `_downstream_head()` 得到初始化点图 | 将同一帧作为图像对输入，经 Pi3X decoder 与几何 heads，取该帧的局部点图及 sigmoid 后的置信度 |
| `match_pair(frame_i, frame_j)` | 两帧联合解码，输出同一参考坐标系中的点图和描述子，再通过点图与描述子匹配生成索引 | 联合预测两帧局部点图与 c2w 位姿，将点变换到另一相机系后投影，检查边界、深度与三维距离，生成对应索引 |
| `match_symmetric_batch(...)` | 对图像对按两种顺序解码，生成双向匹配，用于视觉因子构建 | 对每个图像对推理，从同一组局部点图与位姿生成两个方向的几何对应 |
| `infer_window(frames)` | adapter 逐帧调用单帧推理并汇总点图 | 组合窗口图像与编码特征，一次多帧解码得到各帧点图、置信度和位姿 |
| `build_pair_constraints_from_window(frames, edges)` | 原图像对流程逐边调用匹配接口构建约束 | 延迟路径从一次窗口推理中，为指定边生成双向索引、有效掩码与单帧置信度 |

`match_pair()` 用 `PairMatchResult` 返回匹配索引 `idx_i2j`、有效掩码 `valid_match_j`、点图 `Xii/Xji` 和置信度。`Xii` 是帧 i 的局部点图，`Xji` 是帧 j 的点变换到帧 i 相机系后的结果；`idx_i2j` 按帧 j 像素排列，数值指向帧 i 的像素。MASt3R 的 `C/Q` 分别来自点图和描述子置信度；Pi3X 使用点图置信度适配这些字段，双向约束保留单帧值，由因子图组合后筛选。

因此，替换模型主要改变编码、几何预测与匹配生成方式。原有后端仍消费图像对约束；Pi3X 的多帧能力通过窗口接口提供给延迟更新路径。

接口与模型选择见 [base.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/frontend_model/base.py)、[factory.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/frontend_model/factory.py)；MASt3R 封装见 [mast3r_adapter.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/frontend_model/mast3r_adapter.py)，Pi3X 封装见 [pi3_adapter.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/frontend_model/pi3_adapter.py)、[pi3x_utils.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/frontend_model/pi3x_utils.py)和 [pi3_matching.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/frontend_model/pi3_matching.py)。

### 2. 简单光流关键帧选择

原始 MASt3R-Fusion 逐帧调用前馈模型，与最近关键帧进行密集匹配和位姿跟踪，再根据匹配情况选择关键帧。改造后，逐帧先用 Shi-Tomasi 特征点与 LK 光流跟踪，依据跟踪点数量及比例、视差和帧间隔决定是否插入关键帧，将关键帧选择所需的信息交给轻量跟踪提供。IMU 的旋转与平移信息参与触发：小运动片段抑制冗余关键帧，大旋转或达到最大间隔时触发插入。

光流提供关键帧选择所需的运动与跟踪信息，选出的关键帧再进入 Pi3X 视觉更新。当前 SparseMap 还维护稀疏地图，通过三角化与 PnP 提供逐帧位姿，并在后端更新后对齐到最新关键帧。

实现见 [sparse_flow_frontend.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/sparse_flow_frontend.py)和 [sparse_map.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/sparse_map.py)，LK 实现参考 [pySLAM](https://github.com/luigifreda/pyslam)。

### 3. 后端优化：延迟视觉更新与 IMU 重积分

原始 MASt3R-Fusion 在新关键帧加入后，构建其与相邻及检索关键帧的图像对视觉因子，并更新滑窗。改造后，将视觉推理与逐帧状态传播分开：视觉惯性初始化前，新关键帧仍按图像对更新；初始化后，利用 IMU 传播窗口间的状态，累计 `delayed_batch_keyframes` 个关键帧，再连同相邻前驱组成窗口，执行一次 Pi3X 多帧推理。当前延迟路径由窗口输出构建相邻关键帧约束。

窗口输出生成相邻关键帧的双向匹配，视觉点图对齐形成 Hessian 因子，与 IMU 因子共同优化位姿、尺度、速度和 bias。沿用原后端的 IMU 预积分机制，按关键帧时间区间读取原始测量，以当前 bias 重新积分，使 IMU 因子与延迟视觉窗口对应。边缘化调整到窗口优化完成之后，并按全局关键帧索引回写状态；缓冲区滚动前处理剩余关键帧。

已有关键帧点图在首次视觉初始化后保持固定，后续窗口几何用于构造匹配，保持后端点图锚点的一致性。窗口调度见 [main_pi3x_delayed.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/main_pi3x_delayed.py)，优化、重积分与边缘化见 [pi3x_global_opt.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/pi3x_global_opt.py)。

## 数据组织

输入为带时间戳的图像与 IMU，另提供相机内参、畸变和相机—IMU 外参。建议按序列组织：

```text
sequence/
├── images/
│   ├── 000000.jpg
│   └── 000001.jpg
├── camstamp.txt
└── imu.txt
```

`camstamp.txt` 每行记录时间戳与相对图像目录的文件名，时间单位为秒：

```text
1710000000.000 000000.jpg
1710000000.050 000001.jpg
```

`imu.txt` 使用无表头的七列数据，可用空格或逗号分隔，列顺序为：

```text
timestamp gx gy gz ax ay az
```

时间戳使用秒；角速度单位由 `ms_opt.imu_format` 指定：`custom_deg` 为 °/s，`custom_rad` 为 rad/s；加速度使用 m/s²。图像时间戳与 IMU 使用同一时间基准，IMU 时间严格递增，并覆盖处理的图像区间。`--imu_dt` 加到 IMU 时间戳上，用于相机与 IMU 的固定时间偏移。

标定文件包含原图尺寸、内参与畸变参数，以及 `Tic` 外参，格式参考 [intrinsics_kitti360.yaml](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/config/intrinsics_kitti360.yaml)。图像缩放、裁剪时同步更新内参。读取逻辑见 [dataloader.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/dataloader.py)和 [data_utils.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/geoFunc/data_utils.py)。

## 运行入口

`main.py` 保留逐帧/图像对入口；Pi3X 延迟匹配使用 `main_pi3x_delayed.py`。环境安装见 [MASt3R-Fusion README](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/README.md)，Pi3X 权重默认放在 `checkpoints/pi3x/model.safetensors`。

运行配置为 [base_kitti360_pi3x_delayed.yaml](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/config/base_kitti360_pi3x_delayed.yaml)，继承 [Pi3X 配置](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/config/base_kitti360_pi3x.yaml)。使用时确认数据与标定、`frontend_model.weights`、图像尺寸、`delayed_batch_keyframes` 和 IMU 格式。

从 MASt3R-Fusion 仓库根目录运行，数据与标定路径替换为对应序列：

```bash
python main_pi3x_delayed.py \
  --config config/base_kitti360_pi3x_delayed.yaml \
  --calib config/intrinsics_kitti360.yaml \
  --dataset /path/to/sequence/images \
  --stamp_path /path/to/sequence/camstamp.txt \
  --imu_path /path/to/sequence/imu.txt \
  --imu_dt -0.04 \
  --frontend-weights checkpoints/pi3x/model.safetensors \
  --result_path result_pi3x_delayed.txt --save_h5 --no-viz
```

`-0.04` 为 KITTI-360 示例偏移，其他序列使用对应值。`--start_from`、`--end_at` 控制处理范围。运行保存位姿结果，启用 `--save_h5` 后保存 `data.h5`，同时输出 `graph.pkl`；窗口位姿诊断写入 `result_path.pi3x_pose_debug.csv`。位姿结果使用独立文件名，H5 与因子图写入运行目录，实验完成后与实际配置一起归档。

## Tips：Foxglove 可视化调试

加入了 Foxglove WebSocket 发布，可在无桌面的服务器上观察当前图像、位姿、轨迹、关键帧点云、稀疏地图与图边。运行命令追加 `--foxglove`，默认监听 `127.0.0.1:8765`。

本地连接远程服务器时，先建立端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

打开 [Foxglove 网页端](https://app.foxglove.dev/)，通过 Foxglove WebSocket 连接 `ws://127.0.0.1:8765`，选择对应图像与 3D 通道。`--foxglove-hz` 控制发布频率。实现见 [foxglove_debug.py](https://github.com/p-xiexin/MASt3R-Fusion/blob/pi3x-delayed-matching/mast3r_fusion/foxglove_debug.py)。
