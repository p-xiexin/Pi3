# ABotRecon 上的流式 Glob3R 设计

第一轮使用已知相机内参，将现有 Glob3R 匹配分支接到 ABotRecon 的因果几何特征上。每张图像只经过一次几何主干。时刻 $t$ 固定以上一帧 $I_{t-1}$ 为 reference、当前帧 $I_t$ 为 target，输出一组 $t-1\rightarrow t$ 的稠密 warp 和匹配置信度。本文确定这一步的网络接口、流式缓存、监督信号和训练验证。位姿与尺度优化、冻结匹配头后的联合 refine 保留为后续独立验证目标。

## A 网络结构与流式状态

### A1 符号和相机约定

输入为按到达顺序排列的图像流 $\{I_t,K_t^{\mathrm{src}}\}_{t\geq0}$。$t$ 是处理顺序，原始帧号和时间戳单独保存。第一帧确定世界坐标系。所有外部位姿均采用 camera to world 约定

$$
T_t=\begin{bmatrix}R_t&C_t\\0&1\end{bmatrix},\qquad
T_{j\leftarrow i}=T_j^{-1}T_i,\qquad
\Delta T_t=T_{t-1}^{-1}T_t.
\tag{A1}
$$

$C_t$ 表示世界坐标中的相机中心。像素坐标从零开始，$u=(x,y)^\top$，$\bar u=(x,y,1)^\top$。$\pi_K(X)$ 表示先乘内参矩阵再做透视除法。用于坐标回归的投影须满足有限值和正深度条件，置信度标签的有效性由 B2 定义。

| 符号 | 含义 | 单帧张量形状 |
| --- | --- | --- |
| $B$ | batch 中独立序列的数量 | 在线运行取 1 |
| $H,W$ | 处理后的图像高和宽 | 初始设置为 280 和 504 |
| $h,w,M$ | patch 高、宽和数量 | $20,36,720$ |
| $C$ | DINO 与几何 decoder 通道数 | 1024 |
| $G_t$ | 最后两层几何特征的拼接 | $B\times M\times2048$ |
| $E_t^\ell$ | 四层 DINO patch 特征 | 每层 $B\times M\times1024$ |
| $Z_t$ | matching decoder 输出 | $B\times M\times1024$ |
| $P_t$ | ABotRecon 原始局部点图 | $B\times H\times W\times3$ |
| $D_t$ | 原始点图的第三个分量 | $B\times H\times W$ |
| $W_t^{(s)}$ | 上一帧网格上的当前帧对应坐标 | $B\times2\times H_s\times W_s$ |
| $l_t^{(s)}$ | 上一帧网格上的匹配置信度 logit | $B\times1\times H_s\times W_s$ |

其中 $H_s=\lfloor H/s\rfloor$，$W_s=\lfloor W/s\rfloor$。本文统一令 $W_t=W_{t-1\rightarrow t}$，$l_t=l_{t-1\rightarrow t}$。source 与 reference 均表示 $t-1$，target 表示 $t$。warp 和置信度的定义域均为上一帧网格，warp 的值域为当前帧图像坐标。每个时刻只计算这一个方向。原始帧号随输出保存，用于核对相邻配对。第一帧建立缓存并输出原始几何，从第二帧开始输出匹配。

### A2 图像变换与已知内参

首轮 ScanNet 训练复用数据集预处理，将 RGB 调整到深度图分辨率，执行主点周围的裁剪、等比例缩放和最终裁剪，得到高 280、宽 504 的输入。监督使用数据集返回的深度和内参。令处理前后的焦距为 $f_x^{\mathrm{src}},f_y^{\mathrm{src}}$ 与 $f_x,f_y$，主点为 $c_x^{\mathrm{src}},c_y^{\mathrm{src}}$ 与 $c_x,c_y$，定义 $s_x=f_x/f_x^{\mathrm{src}}$、$s_y=f_y/f_y^{\mathrm{src}}$。等效坐标变换为

$$
A_t=\begin{bmatrix}
s_x&0&c_x-s_xc_x^{\mathrm{src}}\\
0&s_y&c_y-s_yc_y^{\mathrm{src}}\\
0&0&1
\end{bmatrix},\qquad K_t=A_tK_t^{\mathrm{src}}.
\tag{A2}
$$

主点变化包含数据集实际执行的裁剪偏移和像素中心修正。GT 投影直接使用处理后的 $K_t$。输入须为已去畸变图像及其对应内参，或由输入模块同时完成去畸变和内参更新。第一轮固定焦距、主点和畸变。

ScanNet 深度从毫米转换为米，重采样沿用数据集的最近邻插值，零深度保持无效。RGB 缩小时采用 Lanczos，放大时采用双三次插值。最终有效掩码同时检查深度和反投影点的有效性。

### A3 ABotRecon 特征接口

ABotRecon 保留 DINOv2 ViT-L/14、36 个交替的单帧与因果跨帧 decoder block、五个特殊 token，以及原有局部点图和相邻位姿分支。使用 decoder 最后两个 block 的输出拼接得到 $G_t$，去除特殊 token 后交给匹配分支。DINO 特征取从零开始的第 5、11、17、23 层，执行现有 encoder norm 并去除 DINO 特殊 token。

新增的公共分解接口为

$$
(G_t,\{E_t^\ell\},\mathcal M_t)
=\operatorname{encode\_step}(I_t,\mathcal M_{t-1}),
\qquad
(P_t,\Delta T_t,\mathcal C_t)
=\operatorname{geometry\_step}(G_t,\mathcal C_{t-1}).
\tag{A3}
$$

$\mathcal M_t$ 保存主干各跨帧 attention 层最近最多 11 帧的 Key 和 Value，以及时间位置索引，供下一帧读取历史图像信息。

$\mathcal C_t$ 保存第 $t$ 帧的相机描述符、相机分支的图像特征，以及 temporal rotation refiner 最近最多 10 个相邻帧对的特征缓存，供下一帧预测相对位姿并修正旋转。

### A4 KV 缓存与时间位置

第 $\ell$ 个因果跨帧 block 在时刻 $t$ 使用

$$
\mathcal J_t=\{\max(0,t-K+1),\ldots,t\},\qquad
O_t^\ell=\operatorname{Attention}
\left(Q_t^\ell,\operatorname{cat}_{j\in\mathcal J_t}K_j^\ell,
\operatorname{cat}_{j\in\mathcal J_t}V_j^\ell\right),\quad K=12.
\tag{A4}
$$

推理时每层最多保留 11 个历史帧的 KV，在当前帧计算期间临时拼入第 12 帧。历史 KV 保持其生成时的内容。多层因果传播使最终特征可以包含超过十二帧的间接历史信息，重新编码最近十二帧会改变这些历史条件。

RoPE 使用单调递增的处理索引。环形缓存回绕只改变存储位置，不重置时间位置。时间频率支持按索引生成，不能让长度 4096 的预计算表成为长序列上限。单帧 attention、门控投影、特殊 token 空间位置和归一化顺序保持现有语义。

`reset_stream` 同时清空 KV、上一帧相机描述符、rotation refiner 缓冲和上一帧匹配缓存。不同 batch 元素的序列状态独立。第一帧输出单位位姿和空匹配结果。断流后从空状态建立新序列。

### A5 单帧 matching decoder

沿用当前五层 matching decoder。首先进行 $2048\to1024$ 的线性投影，每层使用 16 个 attention head 和 expansion ratio 为 4 的 MLP

$$
Z_t^{(0)}=W_{\mathrm{in}}G_t+b_{\mathrm{in}},\qquad
\widetilde Z_t^{(q)}=Z_t^{(q-1)}+
\operatorname{MHA}\bigl(\operatorname{LN}(Z_t^{(q-1)})\bigr),
\tag{A5}
$$

$$
Z_t^{(q)}=\widetilde Z_t^{(q)}+
\operatorname{MLP}\bigl(\operatorname{LN}(\widetilde Z_t^{(q)})\bigr),
\quad q=1,\ldots,5,\qquad Z_t=\operatorname{LN}(Z_t^{(5)}).
\tag{A6}
$$

每层只处理同一帧的 $M$ 个 patch。跨帧信息由 $G_t$ 提供，matching decoder 保持现有单帧结构。推理时缓存生成后的 $Z_{t-1}$，与当前 $Z_t$ 配对。$G_{t-1}$ 仅依赖截至上一帧的输入，$G_t$ 可以读取截至当前帧的因果状态。训练使用相同的前缀条件。

### A6 相邻帧匹配与坐标嵌入

reference patch 编号为 $n$，target patch 编号为 $m$。对上一帧和当前帧计算

$$
L^t_{nm}=\frac{\langle\widehat Z_{t-1,n},\widehat Z_{t,m}\rangle}{\tau},
\qquad
A^t_{nm}=\frac{\exp L^t_{nm}}{\sum_{m'}\exp L^t_{nm'}},
\qquad \tau=0.1.
\tag{A7}
$$

$\widehat Z$ 是沿通道维做 L2 归一化的 token。$L$ 为交叉熵输入，$A$ 为行归一化概率。softmax 只执行一次，归一化维度始终是 target patch 维。

第一轮保持现有 Fourier 坐标基。对于 target patch 网格坐标 $(m_x,m_y)$，使用

$$
q_m=\left(\frac{2m_x}{w-1}-1,\frac{2m_y}{h-1}-1\right),\quad
\chi_m=\cos(2\pi\Omega q_m)\oplus\sin(2\pi\Omega q_m),\quad
\Omega\in\mathbb R^{512\times2},\quad\Omega_{ab}\sim\mathcal N(0,1).
\tag{A8}
$$

$\Omega$ 使用 seed 0 生成并写入 checkpoint。它是位置编码基，像素 warp 的归一化另由 A11 定义。两者分别保持现有 checkpoint 语义。

$$
F_{t,n}=\operatorname{Linear}_{2048\to1024}
\left(Z_{t-1,n}\oplus\sum_m A^t_{nm}\chi_m\right).
\tag{A9}
$$

$F_t$ 和 DPT 的四层 skip feature 都位于上一帧的 reference 网格。相似度矩阵的行对应上一帧 patch，列对应当前帧 patch。Fourier 聚合使用当前帧坐标基，DPT 的 skip feature 使用上一帧特征，沿用现有 Glob3R 的匹配方向。

### A7 Coarse DPT 与 warp 表示

DPT 将四层 source DINO 特征分别投影至 $[256,512,1024,1024]$ 通道，再变换到 256 个 scratch 通道和 stride $[4,8,16,32]$ 的网格。逐级上采样融合，加入投影后的 $F_t$，输出三个通道

$$
(\widehat W_t^{(4)},l_t^{(4)})
=\operatorname{DPT}(F_t,E_{t-1}^5,E_{t-1}^{11},E_{t-1}^{17},E_{t-1}^{23}),
\qquad p_t^{(4)}=\operatorname{sigmoid}(l_t^{(4)}).
\tag{A10}
$$

warp 的两个通道直接回归归一化 target 坐标，不经过 sigmoid 或坐标范围截断。对完整 target 图像中的像素 $(x,y)$，坐标转换为

$$
\mathcal N_{H,W}(x,y)=
\left(\frac{2(x+1/2)}{W}-1,\frac{2(y+1/2)}{H}-1\right),\qquad
\mathcal N_{H,W}^{-1}(a,b)=
\left(\frac{W(a+1)}2-\frac12,\frac{H(b+1)}2-\frac12\right).
\tag{A11}
$$

warp 的定义域是 source 网格，值域始终指向完整 target 图像。改变输出网格分辨率时只重采样定义域，不额外乘 stride。`grid_sample` 使用 `align_corners=False`。DPT 内部 skip feature 的插值保留现有 `align_corners=True`，这项内部插值不改变 A11 的 warp 值域。warp 输出与置信度 logit 输出使用 FP32。

### A8 匹配细化网络

细化分支保留 VGG19-BN 前三层级的图像特征。原始通道 $[256,128,64]$ 分别投影到 stride $[4,2,1]$ 的 $[192,48,12]$ 通道。每张新增图像只计算一次特征金字塔。

| stride $s$ | 图像特征通道 $C_s$ | displacement 通道 | correlation 通道 | refiner 输入通道 |
| --- | --- | --- | --- | --- |
| 4 | 192 | 79 | 49 | 512 |
| 2 | 48 | 23 | 9 | 128 |
| 1 | 12 | 8 | 0 | 32 |

每一级先将前一级 warp 和 logit 停止梯度并插值到当前 source 网格。$g_s$ 为当前网格的归一化像素中心，$S=\operatorname{diag}(W/512,H/512)$。细化输入为

$$
U_s=\phi_{t-1}^s\oplus
\operatorname{sg}\bigl(\phi_t^s(\widehat W)\bigr)\oplus
\operatorname{Conv}_{1\times1}\bigl(S(\widehat W-g_s)\bigr)\oplus
\operatorname{sg}\bigl(\operatorname{Corr}_s(\phi_{t-1}^s,\phi_t^s,\widehat W)\bigr).
\tag{A12}
$$

correlation 在 stride 4 和 2 使用半径 3 和 1，stride 1 不使用 correlation。每个 refinement block 包含一个输入单元和八个同通道单元，每单元依次执行 $5\times5$ depthwise convolution、BatchNorm、ReLU 和 $1\times1$ pointwise convolution，最后输出三个残差通道。

$$
(d_s,\delta l_s)=\operatorname{Refine}_s(U_s),\qquad
\widehat W^{(s)}=\widehat W+\frac{d_s}{4(W_s,H_s)},\qquad
l^{(s)}=l+\delta l_s.
\tag{A13}
$$

式 A12 与 A13 保持当前 RoMaV2 兼容的梯度截断和残差尺度。上一帧的 source 特征参与梯度，当前帧的 target 采样和 correlation 分支停止梯度。

VGG 与 refinement block 中所有 BatchNorm 的 running mean 和 variance 在训练、验证、推理中保持固定，affine 参数按训练阶段决定是否更新。由预训练权重初始化统计量。这样同一 batch 的未来帧或其他时刻的 pair 不会通过批统计修改历史特征。单独关闭 running statistics 更新但仍使用当前 batch statistics 不满足此要求。

### A9 上一帧缓存与相邻配对

| 状态 | 内容 | 保留范围 |
| --- | --- | --- |
| 主干状态 | 18 个跨帧 block 的历史 KV | 每层最近 11 帧 |
| 相机状态 | 上一帧描述符、patch 特征及旋转修正缓冲 | 原相机头所需范围 |
| 匹配状态 | 上一帧的 $Z_{t-1}$、四层 $E_{t-1}^{\ell}$、$\phi_{t-1}^s$ 和有效图像区域 | 上一帧 |

时刻 $t$ 生成当前帧特征后，读取上一帧匹配缓存并计算 $W_t$ 和 $p_t$。计算完成后用当前帧特征替换缓存。持久匹配缓存只有一帧，执行期间同时持有上一帧和当前帧特征。完整相关矩阵只用于当前匹配和训练损失，推理结束后释放。

整段训练先按因果顺序生成 ABotRecon 特征，再将相邻特征组织为两帧样本

$$
\mathcal P=\{(G_{t-1},G_t)\}_{t=1}^{L-1}.
\tag{A14}
$$

$L$ 是 clip 长度。每个样本内部 reference 索引为 0，target 索引为 1。四层 DINO 特征与细化图像特征采用相同配对。现有匹配头会将 reference 之外的所有输入帧视为 target，因此调用时只传入已选中的这两帧。ABotRecon 的因果编码在整段序列上进行，其历史状态持续保留。

## B 匹配监督与训练协议

### B1 因果训练样本

每个样本来自一条具有真实时间顺序的序列。随机选择起点后按时间顺序输入，reference 始终为当前处理时刻的上一帧。原始帧号和处理索引同时保留，第一轮输入时间 stride 为 1。

预训练 clip 长度从 $\{32,64,128\}$ 选择，概率分别为 $\{0.5,0.25,0.25\}$。每个 clip 最多采样四个相邻帧对，末帧对应的帧对固定保留，其余从有效时刻均匀抽取。启动阶段的相邻帧对也属于可采样集合。整段图像按顺序经过主干，抽样只减少参与匹配训练的帧对数量。

冻结的 ABotRecon 按 A4 生成因果特征。对选中帧对涉及的帧计算 matching token 和细化特征，最多涉及八帧。coarse 阶段对两帧的 matching token 保留梯度，refinement 阶段沿用 A8 的梯度截断。推理缓存中的 detached matching token 不能替代训练所需的 reference 梯度路径。每次 optimizer update 后重新生成可训练匹配特征。

随机起点定义新的训练子序列，其状态从空缓存开始。验证同时覆盖启动阶段和连续运行阶段。长序列测试从序列起点连续积累因果状态。

### B2 几何真值对应关系

监督使用 GT 深度、GT 位姿和已知内参。所有数据位于 A2 定义的处理后图像坐标系。对上一帧的 source 像素 $u$，计算

$$
X_{t-1}^*(u)=D_{t-1}^*(u)K_{t-1}^{-1}\bar u,\qquad
Y_t^*(u)=(T_t^*)^{-1}T_{t-1}^*\overline{X_{t-1}^*(u)},\qquad
W_t^*(u)=\pi_{K_t}(Y_{t,1\ldots3}^*(u)).
\tag{B1}
$$

$z_t^*(u)$ 为变换后的深度。$\widetilde D_t^*(u)$ 为当前帧 GT 深度在 $W_t^*(u)$ 处的采样值，沿用现有 GT warp 构造的深度采样方式。令 $\mathcal D_t=[0,W-1]\times[0,H-1]$，定义

$$
b_t(u)=\mathbf1[W_t^*(u)\in\mathcal D_t],\qquad
h_t(u)=\mathbf1[z_t^*(u)>0,\ \widetilde D_t^*(u)>0,\
z_t^*(u)\text{ 和 }\widetilde D_t^*(u)\text{ 均有限}].
\tag{B2}
$$

$v_t(u)$ 表示额外的数据有效性掩码。它要求上一帧像素位于有效图像区域，GT 深度有限且为正，GT 位姿有效，变换和投影有限，且该点属于允许刚体监督的静态区域。对于图内投影，还要求 target 位置位于有效图像区域及允许监督的区域。填充区域和无法确定运动的动态区域取零。

沿用 Glob3R 的深度一致性判据和置信度监督掩码

$$
y_t(u)=v_t(u)b_t(u)h_t(u)
\mathbf1\left[
\frac{|\widetilde D_t^*(u)-z_t^*(u)|}{\widetilde D_t^*(u)}<0.05
\right],\qquad
m_t(u)=v_t(u)\mathbf1[(b_t(u)=1\land h_t(u)=1)\lor b_t(u)=0].
\tag{B3}
$$

相对深度误差只在 $b_t h_t=1$ 时计算。坐标回归和 patch 分类使用 $m_t y_t=1$ 的正样本，置信度损失使用 $m_t=1$ 的样本。图内且有有效 target 深度的遮挡点为置信度负样本，有限的越界投影也为负样本。图内缺失 target 深度的位置跳过监督。稀疏深度的无效区域保留为未知标签，不补充推断深度作为 GT。

### B3 Patch 分类与 dense warp 损失

第一轮保留当前 `patch_nll_targets` 的离散化。GT warp 和正样本掩码以最近邻缩放到 $h\times w$，target 像素坐标量化为

$$
m_x^*=\operatorname{round}\left(\frac{x^*}{W-1}(w-1)\right),\quad
m_y^*=\operatorname{round}\left(\frac{y^*}{H-1}(h-1)\right),\quad
m^*=m_y^*w+m_x^*.
\tag{B4}
$$

该离散化与式 A8 的 endpoint 坐标基沿用现有实现，本轮保持其数值定义。$\Omega_{\mathrm{patch}}^t$ 为最近邻采样后满足 $m_t y_t=1$ 且 target patch 索引有效的 reference patch 集合。

$$
\mathcal L_{\mathrm{patch}}^t
=-\frac1{|\Omega_{\mathrm{patch}}^t|}
\sum_{n\in\Omega_{\mathrm{patch}}^t}\log A^t_{n,m_n^*}.
\tag{B5}
$$

对每个预测层级，将 GT warp 最近邻采样到 source 输出网格，再用 A11 转换为 target 归一化坐标。有效 warp 监督集合为 $\Omega_s^t=\{u\mid m_t(u)y_t(u)=1\}$，掩码按同一 source 网格最近邻采样

$$
\mathcal L_{\mathrm{warp},s}^t
=\frac1{|\Omega_s^t|}\sum_{u\in\Omega_s^t}
\left(\|\widehat W_t^{(s)}(u)-\mathcal N_{H,W}(W_t^*(u))\|_2^2
+\epsilon^2\right)^{\gamma/2},\qquad
\epsilon=10^{-3},\quad\gamma=0.5.
\tag{B6}
$$

令 $\Gamma_s^t=\{u\mid m_t(u)=1\}$，掩码按同一 source 网格最近邻采样，使用 FP32 BCE with logits

$$
\mathcal L_{\mathrm{conf},s}^t
=\frac1{|\Gamma_s^t|}\sum_{u\in\Gamma_s^t}
\left[\operatorname{softplus}(l_t^{(s)}(u))-y_t(u)l_t^{(s)}(u)\right].
\tag{B7}
$$

当一个像素集合为空，对应损失返回与预测图连接的零。在 B1 选中的时刻内，令 $\mathcal T_p$、$\mathcal T_{w,s}$ 和 $\mathcal T_{c,s}$ 分别表示 patch、层级 $s$ 的 warp 和 confidence 具有非空监督集合的时刻。每项损失先在单个相邻帧对的有效像素上平均，再在各自的有效时刻上平均，最后平均 batch 中各条有效序列。空时刻集合的均值定义为零。

$$
\mathcal L_{\mathrm{match}}=
\lambda_p\operatorname{mean}_{t\in\mathcal T_p}\mathcal L_{\mathrm{patch}}^t+
\operatorname{mean}_{s\in\mathcal S}
\left(
\lambda_w\operatorname{mean}_{t\in\mathcal T_{w,s}}\mathcal L_{\mathrm{warp},s}^t
+\lambda_c\operatorname{mean}_{t\in\mathcal T_{c,s}}\mathcal L_{\mathrm{conf},s}^t
\right).
\tag{B8}
$$

coarse 阶段取 $\lambda_p=\lambda_w=\lambda_c=1$，$\mathcal S$ 仅包含 coarse 输出。细化阶段取 $\lambda_p=0$、$\lambda_w=\lambda_c=1$，$\mathcal S$ 包含三个可训练的细化输出，冻结 coarse 的三项损失另作监控。匹配置信度学习可见性与几何有效性，不将其解释为已标定的坐标误差方差。

### B4 预训练阶段和数值设置

| 项目 | Coarse 预训练 | 匹配细化训练 |
| --- | --- | --- |
| ABotRecon | 全部冻结并置于 eval | 全部冻结并置于 eval |
| matching decoder 与 DPT | 更新 | 冻结 |
| VGG、projection、refinement | 不参与 forward | 更新参数，固定全部 BN 统计量 |
| 初始化 | 已有兼容 Glob3R 权重或随机初始化 | 本阶段 coarse checkpoint 与兼容 RoMaV2 细化权重 |
| optimizer | AdamW | AdamW |
| 峰值学习率 | $10^{-4}$ | $5\times10^{-5}$ |
| epoch 数与每轮 iter 数 | 80 与 800 | 80 与 800 |
| optimizer updates | 32000 | 32000 |
| warmup updates | 2000 | 2000 |
| weight decay | 0.05，bias 与 norm 参数不衰减 | 相同 |
| betas | $(0.9,0.95)$ | 相同 |
| 梯度裁剪 | 全局范数 1 | 全局范数 1 |
| 每 rank clip 数与累积步数 | 1 与 2 | 1 与 2 |

训练采用 Glob3R 附录 B 的 coarse 与 refinement 分阶段方式。表中学习率和训练长度为流式实验的初始设置。溢出跳过的更新不推进 scheduler。有效 clip batch 为 world size 乘每 rank clip 数乘累积步数。

训练长度由 `train.num_epoch` 和 `train.iters_per_epoch` 控制。一个 iter 对应每 rank 的一次 microbatch 前向与反向，`train.gradient_accumulation_steps` 个 iter 构成一次参数更新。默认设置共有 64000 个 iter，在无溢出跳步时完成 32000 次更新。代码与 TensorBoard 的 `global_step` 只统计成功更新。`train.lr_scheduler.warmup_steps` 以参数更新计数，共享学习率调度器的索引以 microbatch 计数，每次成功更新推进一个完整的累积组。

每个 epoch 结束后验证，`test.iters_per_test` 限制验证批数，取负一时遍历验证集。`train.print_freq` 与 `test.print_freq` 以各自循环的 iter 计数，`log.summary_interval` 以成功更新计数，`log.ckpt_interval` 以 epoch 计数。日志使用 `Epoch`、`Validation Epoch`、`loss_nll`、`loss_warp` 和 `loss_match_confidence`。每轮 iter 数须能整除梯度累积步数，保证 epoch checkpoint 不遗漏待累积的梯度。

主干和 attention 默认 BF16，warp 输出、logit 和 loss reduction 使用 FP32。训练 seed 为 42，数据基础 seed 为 666。首轮关闭图像增强，保留数据集确定性的裁剪与缩放。

### B5 训练数据与权重兼容性

首轮匹配训练与验证仅使用 ScanNet，复用 ABot 的 ScanNet 数据集定义、场景划分和按时间排序的抽帧方式。相邻帧指排序后的相邻输入，原始帧号可以有间隔。匹配训练覆盖片段长度与分辨率，禁用 wrapper 的折返与重复采样，关闭图像增强，RGB 保持在 0 到 1。超参数选择使用验证集，数据集扩充留到本轮验证完成后。

coarse 与 refinement 使用相同训练清单和验证划分。多域训练同时统计样本数量和有效匹配数，以衡量稀疏深度域提供的监督量。

匹配权重采用式 A11 的归一化坐标输出。初始化权重须与当前 coarse 和 refinement 的坐标语义及层形状兼容。

## C 推理与匹配验证

### C1 逐帧执行

1. 读取当前 RGB 和已知内参，完成 A2 的图像变换与有效区域生成。
2. 执行一次 ABotRecon 编码，更新主干和相机状态，得到当前原始几何及匹配所需特征。
3. 计算当前 matching token，启用 refinement 时同时计算细化金字塔。
4. 从第二帧起，以上一帧缓存为 reference、当前帧为 target，输出一组 warp 和 confidence。
5. 用当前帧特征替换上一帧匹配缓存，释放当前相关矩阵，返回当前原始几何和相邻匹配结果。

所有步骤只访问当前和过去输入。全 clip 训练中的向量化计算也必须满足相同的前缀依赖。替换未来帧时，之前时刻的匹配结果应保持不变。

### C2 匹配对照与精度

对照使用相同的 ABotRecon 因果特征、图像预处理、已知内参和相邻帧对。

| 配置 | 比较内容 |
| --- | --- |
| 兼容的已有 Glob3R 权重直接接入 | 相邻匹配的初始化基线 |
| 相邻帧监督训练后的 coarse | 与初始化 coarse 比较匹配精度 |
| 相邻帧监督训练后的 refinement | 与初始化的同级 refinement 比较最终精度，并报告相对 coarse 的增益 |

在 B2 定义的有效静态 GT 正样本上报告 EPE 和 PCK 1／3／5 像素。各层 warp 均先还原为完整图像像素坐标。按序列分别统计，再汇总序列结果。

匹配置信度报告 BCE 及阈值下的匹配精度和覆盖率。阈值采用 $\{0.25,0.5,0.75\}$，像素正确性阈值采用 $\{1,3,5\}$。可评价集合取 $m_t=1$ 的位置，选中集合为其中 $\hat p_t\geq\tau_p$ 的位置，正确匹配要求 $y_t=1$ 且坐标误差不超过像素阈值。precision 的分母为选中数量，覆盖率的分母为全部 GT 正样本数量，分子均为正确匹配数量。空选中集合的 precision 记为未定义并同时报告选中数量。

分别报告启动阶段和连续运行阶段。相邻配对始终使用实际输入顺序中的前后两帧。首轮采用相同的固定 seed 和验证序列比较各配置。

### C3 时间与缓存

记录单帧编码、几何头、matching decoder、细化金字塔、相邻匹配及端到端延迟，报告中位数和 95 分位数。GPU 计时包含同步操作。

检查持续运行时的缓存容量。主干每层保留最近 11 帧 KV，匹配分支持久保留上一帧特征，匹配执行期间最多同时持有两帧特征。每个有效时刻只计算一个 $M\times M$ 相关矩阵，匹配计算复杂度为 $O(M^2C)$。记录实际峰值显存，验证长序列下活跃缓存不随总帧数增长。

## D 接口与实施顺序

### D1 模块接口

| 模块 | 责任 |
| --- | --- |
| 因果特征接口 | 提供逐帧编码和相机状态更新，保留现有完整 clip 路径 |
| 上一帧缓存 | 保存上一帧匹配特征和帧号，完成替换与重置 |
| 相邻匹配调用 | 将上一帧和当前帧组织为两帧输入，复用现有 Glob3R 匹配头 |
| 相邻监督 | 组织同一对帧的 GT 深度、位姿和内参，计算 B 节三项损失 |
| 训练入口 | 提供 coarse 与 refinement 两个训练阶段 |
| 输出接口 | 返回 reference 与 target 帧号、warp、confidence 及当前原始几何 |

在线输出默认仅保留最终匹配结果，训练和诊断可返回 coarse、各级 refinement 以及相关 logits。公共 warp 使用完整 target 图像中的像素坐标，训练和细化内部使用式 A11 的归一化坐标。状态由调用方显式持有，每条序列独立重置。

### D2 实施与验证

先实现 ABotRecon 的逐帧特征和状态接口，再接入现有 Glob3R 的相邻帧调用。训练时先生成因果主干特征，再构造两帧匹配 batch。GT reference 深度、reference 内参和 reference 位姿均取 $t-1$，target 数据均取 $t$。

先完成 coarse 训练，再加载对应 coarse 权重进行 refinement 训练。匹配头的层结构、Fourier 编码、patch 标签离散化、归一化 warp、细化残差尺度和梯度截断沿用现有实现。

验证覆盖相邻帧索引、输出方向、GT 投影、像素与归一化坐标转换、可见性标签和缺失深度掩码。使用具有已知投影的几何样例检查坐标及 mask，使用真实训练 batch 检查前向、反向和参数更新。

整段因果计算与逐帧计算使用相同精度、eval 模式和初始状态进行比较。检查未来帧扰动、KV 淘汰、上一帧缓存替换、序列重置和长时间位置索引。训练检查各阶段可训练参数的梯度，以及冻结主干、冻结 coarse 和 BN 统计量的状态。

## E 后续验证目标

匹配头训练完成后，分别验证推理时的位姿与尺度优化收益，以及冻结匹配头后的联合 refine 训练收益。近期轨迹允许在固定窗口内修正。后端的约束、变量和优化方式在相邻匹配结果确认后单独讨论，并据此补充设计。

## F 设计依据

Glob3R 论文及附录提供现有匹配分支、几何真值构造、三项监督和 coarse 与 refinement 分阶段训练的依据。本轮沿用当前实现的 source 网格、归一化 warp、单次 softmax 和 patch 离散化约定，将配对统一为上一帧 reference 与当前帧 target。

ABotRecon 提供因果几何特征和逐帧状态。SURE-Map 的位姿与尺度校正思路留待后续几何验证阶段讨论。
