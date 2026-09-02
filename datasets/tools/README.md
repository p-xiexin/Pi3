# Pi3 数据集工具包

这个目录提供一个统一入口，用同一套命令完成数据源查询、容量检查、下载、断点续传、解压、格式转换和目录校验。每个数据集都有两个配方。

`minimal` 也可以写成 `test`。它只准备一次真实加载和可视化所需的数据。

`train` 也可以写成 `full`。它准备官方训练集或当前加载器可直接消费的完整数据根目录。

## 五分钟开始

在仓库根目录运行。

```powershell
conda activate torch-cpu
python -m pip install -r datasets/tools/requirements.txt
python -m datasets.tools list
python -m datasets.tools info sintel
python -m datasets.tools doctor sintel --profile test --proxy localhost:7890 --network
python -m datasets.tools fetch sintel --profile test --proxy localhost:7890
python -m datasets.tools verify sintel
```

Windows 用户也可以使用短入口。`.cmd` 文件不依赖 PowerShell 脚本执行策略。

```powershell
.\datasets\tools\dataset.cmd list
.\datasets\tools\dataset.cmd fetch redwood --profile test --proxy localhost:7890
```

## 常用任务

### 下载一个最小公开样本

```powershell
python -m datasets.tools fetch redwood --profile minimal --proxy localhost:7890
python -m datasets.tools fetch eth3d_slam --profile minimal --proxy localhost:7890
python -m datasets.tools fetch sintel --profile minimal --proxy localhost:7890
python -m datasets.tools fetch tum_rgbd --profile minimal --proxy localhost:7890
python -m datasets.tools fetch hypersim --profile minimal --proxy localhost:7890
```

TUM RGB-D、Redwood、ETH3D SLAM、Hypersim 和 Sintel 会在下载或接入完整训练集后自动生成 `pi3_index.npy`。最小集还会同步生成 `data/dataset_cache/<dataset>_minimal.npy`，供项目配置统一引用。索引保存序列边界、帧路径、相机位姿和数据集特有的内参元数据。训练初始化直接读取索引，避免每个进程重复遍历目录和解析轨迹文本。索引中的路径相对于数据根目录，因此整个目录可以移动。

已有数据可以独立补建或刷新索引。

```powershell
python -m datasets.tools index tartanair
python -m datasets.tools index hypersim --profile full --root D:\datasets\hypersim
python -m datasets.tools index sintel --root D:\datasets\sintel --render-pass final
```

当前加载索引的新增数据集包括 `tum_rgbd`、`redwood`、`eth3d_slam`、`hypersim` 和 `sintel`。TartanAir 已恢复原加载逻辑。ScanNet 与 CO3Dv2 继续使用各自加载器原有的索引或缓存机制。

Hypersim 的 minimal 配方只分块读取 `ai_001_001` 中一个相机的 8 帧 RGB、深度和位姿。完整训练集约 1.9 TB，先检查磁盘，再显式接受许可证并启动官方工具。

```powershell
python -m datasets.tools doctor hypersim --profile full --network
python -m datasets.tools fetch hypersim --profile full --accept-license
```

下载器把归档保存在 `data/.dataset_tools_cache`。重复运行会复用完整归档。已经通过目录校验的 minimal 数据会直接返回。需要重新执行时添加 `--force`。

### 处理一个官方 Waymo TFRecord

先在 Waymo 官网登录并接受许可证，然后下载一个 Perception v1 TFRecord。

```powershell
python -m datasets.tools fetch waymo `
  --profile minimal `
  --input D:\datasets\waymo\segment-example.tfrecord `
  --output data\waymo_processed `
  --accept-license
```

完整训练数据可以传入一个 TFRecord 目录。工具会逐个序列转换为 `WaymoPi3XDataset` 使用的 RGB、投影 LiDAR 深度、内参、相机外参和自车位姿目录。

```powershell
python -m datasets.tools fetch waymo `
  --profile train `
  --input D:\datasets\waymo\training `
  --output D:\datasets\waymo_processed `
  --accept-license
```

### 使用需要账户或官方工具的数据

`list` 中的 `authorized input` 表示官网要求登录、申请访问或接受条款。工具不会保存账号和 cookie。完成官方获取后，把文件或数据根目录交给 `--input`，工具继续做转换和加载器目录校验。

```powershell
python -m datasets.tools info scannet
python -m datasets.tools fetch scannet `
  --profile minimal `
  --input D:\datasets\scannet_exported `
  --accept-license
```

`official tool` 表示数据集作者已经提供下载脚本。ARKitScenes 与 CO3Dv2 配方会缓存官方仓库并执行作者脚本。TartanAir、ETH3D 全量数据等配方在 `info` 中给出选择规则，完成官方脚本后用 `--input` 接入校验。

### 下载完整训练数据

先运行 `doctor`。它会显示预计下载量、展开后的磁盘量和当前磁盘空间。CO3Dv2 全量约 5.5 TB，ASE 全量约 23 TB。完整下载应当先明确训练所需类别、场景和模态。

```powershell
python -m datasets.tools doctor co3dv2 --profile full
python -m datasets.tools fetch co3dv2 --profile full --proxy localhost:7890
```

用 `--dry-run` 可以在下载前查看将执行的官方命令和目标目录。

```powershell
python -m datasets.tools fetch arkitscenes --profile full --dry-run
```

## 每种数据集的下载方式

下面的命令覆盖 `catalog.json` 中登记的全部数据集。`minimal` 用于加载器测试和可视化，`train` 用于训练数据。自动配方会完成下载、解压、后处理和校验。需要官网授权的数据先按说明取得数据，再通过 `--input` 交给工具检查目录并完成必要的转换或索引生成。`--input` 数据会原地使用，不会复制到默认输出目录。

运行大规模下载前建议先检查空间和实际动作。

```powershell
python -m datasets.tools doctor DATASET --profile train --network --proxy localhost:7890
python -m datasets.tools fetch DATASET --profile train --dry-run --proxy localhost:7890
```

### Waymo Open Dataset

先登录 Waymo 下载页并接受许可证。最小测试传入一个 Perception v1 TFRecord，训练处理传入保存训练 TFRecord 的目录。

```powershell
python -m datasets.tools fetch waymo --profile minimal --input D:\datasets\waymo\segment-example.tfrecord --output data\waymo_processed --accept-license
python -m datasets.tools fetch waymo --profile train --input D:\datasets\waymo\training --output D:\datasets\waymo_processed --accept-license
```

### ScanNet v2

先申请 ScanNet 权限，使用官方 `download-scannet.py` 下载 `.sens`，再用官方 `SensorData` 工具导出 `color`、`depth`、`pose` 和 `intrinsic`。`--input` 指向包含场景目录的父目录。

```powershell
python -m datasets.tools fetch scannet --profile minimal --input D:\datasets\scannet_exported --accept-license
python -m datasets.tools fetch scannet --profile train --input D:\datasets\scannet_exported --accept-license
```

### TartanAir

最小配方自动抽取 `carwelding/Hard/P000` 的左目图像、深度和位姿。训练数据先用官方 TartanAir Python 工具选择环境、难度、相机和模态，随后传入下载根目录。全模态数据达到数十 TB，训练下载应显式选择所需内容。

```powershell
python -m datasets.tools fetch tartanair --profile minimal --proxy localhost:7890
python -m datasets.tools fetch tartanair --profile train --input D:\datasets\tartanair
```

### KITTI Raw 与 Depth Completion

登录 KITTI 官网，下载同步 Raw 数据、标定文件和对应的 Depth Completion ground truth。把 raw 与 depth 根目录放在同一个父目录下，再将该父目录传给工具。

```powershell
python -m datasets.tools fetch kitti --profile minimal --input D:\datasets\kitti --accept-license
python -m datasets.tools fetch kitti --profile train --input D:\datasets\kitti --accept-license
```

### ARKitScenes

配方会缓存 Apple 官方仓库并运行 `download_data.py`。最小配方只下载验证序列 `41069021`，训练配方读取官方训练与验证划分，并下载加载器需要的低分辨率 RGB、深度、轨迹和内参。

```powershell
python -m datasets.tools fetch arkitscenes --profile minimal --proxy localhost:7890
python -m datasets.tools fetch arkitscenes --profile train --proxy localhost:7890
```

### Aria Synthetic Environments

在 ASE 页面获取新的 `aria_synthetic_environments_dataset_download_urls.json`，使用官方 `aria_synthetic_environments_downloader` 下载数据。签名地址会过期。最小测试选择 train split 的 scene 0，训练下载应按容量规划选择所需场景，然后把下载根目录传给工具。

```powershell
python -m datasets.tools fetch ase --profile minimal --input D:\datasets\ASE --accept-license
python -m datasets.tools fetch ase --profile train --input D:\datasets\ASE_train --accept-license
```

### nuScenes

最小配方自动下载官方 `v1.0-mini`。完整训练集需要登录 nuScenes 官网，下载并合并解压全部 `v1.0-trainval` 归档，然后传入合并后的根目录。

```powershell
python -m datasets.tools fetch nuscenes --profile minimal --proxy localhost:7890
python -m datasets.tools fetch nuscenes --profile train --input D:\datasets\nuscenes --accept-license
```

### BlendedMVS

最小配方从官方分卷归档中抽取一个真实场景的图像、深度、相机和配对文件。训练数据需要下载 `BlendedMVS.z01` 至 `BlendedMVS.z15` 以及 `BlendedMVS.zip`，使用 7-Zip 解压后传入数据根目录。

```powershell
python -m datasets.tools fetch blendedmvs --profile minimal --proxy localhost:7890
python -m datasets.tools fetch blendedmvs --profile train --input D:\datasets\BlendedMVS
```

### CO3Dv2

配方会缓存 Meta 官方 CO3D 仓库并运行作者下载脚本。最小配方使用官方 single sequence subset。完整配方默认下载全部类别，归档加解压空间需要约 10 TB，可先用 `--dry-run` 查看命令，再按需要修改上游类别选项。

```powershell
python -m datasets.tools fetch co3dv2 --profile minimal --proxy localhost:7890
python -m datasets.tools fetch co3dv2 --profile train --proxy localhost:7890
```

### TUM RGB-D

最小配方自动下载 `freiburg1_desk` 并生成采样索引。训练数据在官方页面选择实验需要的序列，全部解压到同一根目录，再交给工具生成统一索引。

```powershell
python -m datasets.tools fetch tum_rgbd --profile minimal --proxy localhost:7890
python -m datasets.tools fetch tum_rgbd --profile train --input D:\datasets\tum_rgbd
```

### Redwood RGB-D

最小配方自动下载 Open3D 发布的 Redwood 示例。训练数据从 Redwood 项目页下载所需完整场景并解压到同一根目录，工具会检查目录并生成采样索引。

```powershell
python -m datasets.tools fetch redwood --profile minimal --proxy localhost:7890
python -m datasets.tools fetch redwood --profile train --input D:\datasets\redwood
```

### ETH3D SLAM

最小配方自动下载 `plant_1` 的 mono 与 RGB-D 数据。训练数据使用官方 `download_eth3d_slam_datasets.py` 选择 mono 和 RGB-D 数据，解压完成后传入共同根目录。

```powershell
python -m datasets.tools fetch eth3d_slam --profile minimal --proxy localhost:7890
python -m datasets.tools fetch eth3d_slam --profile train --input D:\datasets\eth3d_slam
```

### Hypersim

最小配方只分块读取 `ai_001_001/cam_00` 的八帧 RGB、深度和轨迹。训练配方运行 Apple 官方 Hypersim 下载器，完整数据约 1.9 TB，展开目录建议预留至少 2.5 TB。

```powershell
python -m datasets.tools fetch hypersim --profile minimal --proxy localhost:7890
python -m datasets.tools fetch hypersim --profile train --proxy localhost:7890 --accept-license
```

### MPI Sintel Depth

最小配方只分块读取 `alley_1` 的八帧 final 图像、metric depth 和相机文件。训练配方自动下载并解压完整训练图像以及修正后的 depth 和 camera 归档。

```powershell
python -m datasets.tools fetch sintel --profile minimal --proxy localhost:7890
python -m datasets.tools fetch sintel --profile train --proxy localhost:7890
```

### 校验当前测试数据并运行可视化

```powershell
python -m datasets.tools verify --all
python tests/dataset_viz.py
```

目录校验只检查加载器需要的文件族及最小数量。`tests/dataset_viz.py` 会真实实例化加载器、读取图像和几何数据，并生成汇总渲染结果。两类检查的结论应分开理解。

## 命令说明

```text
list                         列出数据集及 minimal 和 train 获取方式
info DATASET                 显示官方来源、许可证、容量和前置条件
doctor DATASET               检查磁盘空间，可选检查下载地址
fetch DATASET                执行下载、解压、处理和目录校验
verify DATASET               校验一个加载器数据根目录
verify --all                 校验目录中登记的全部 minimal 数据
index DATASET                为已有序列数据生成训练采样索引
catalog                      显示可编辑的数据集目录文件
```

所有 `fetch` 命令支持这些通用参数。

```text
--profile minimal|test|train|full
--output PATH
--cache-dir PATH
--input PATH
--proxy localhost:7890
--workers 8
--accept-license
--dry-run
--force
```

## 数据集目录

配方保存在 `datasets/tools/catalog.json`。下载地址、容量估计、默认输出目录、官方前置条件和文件校验规则都能直接审阅。新增数据集时添加一个目录项，并同时定义 `minimal` 与 `train`。

当前登记 Waymo、ScanNet、TartanAir、KITTI、ARKitScenes、ASE、nuScenes、BlendedMVS、CO3Dv2、TUM RGB-D、Redwood、ETH3D SLAM、Hypersim 和 MPI Sintel Depth。

## 底层工具

统一入口内部复用这些可独立运行的脚本。

```text
download_http_ranges.py                 并行分块和断点续传
extract_remote_zip_entries.py           只下载远程 ZIP 中指定文件
extract_split_zip_entries.py            从分卷 ZIP 中提取指定文件
prepare_waymo_sample.py                  转换 Waymo TFRecord
validate_waymo_processed_sample.py       检查 Waymo 几何并生成覆盖图
prepare_co3dv2_sample.py                 裁剪 CO3Dv2 类别和标注
build_index.py                          生成可移动的训练采样索引
```

这些脚本保留底层接口，日常使用优先选择 `python -m datasets.tools`。
