# Glob3R SfM 与 BA Pipeline

> 论文公式沿用原编号；实现补充公式不编号。

```text
Pi3 推理 → 关键帧选择 → dense warp → multi-view tracks → pose graph
        → rotation averaging → translation averaging → BA → dense reconstruction
```

**1. Pi3 推理。** 输入图像集合，预测相机位姿、局部点图、置信度和近似 metric scale：

$$
f\!\left(\{I_i\}_{i=1}^{N}\right)
=
\{\mathbf T_i,\mathbf X_i,\mathbf C_i,m_i\}.
\tag{1}
$$

当前实现使用 Pi3，无 metric head，取 $m=1$。

**2. 关键帧选择。** 候选帧到已有关键帧的最大有效投影数为

$$
n_t
=
\max_{I_r\in\mathcal K}
\sum_{\mathbf u}
\mathbf 1\!\left[
\pi\!\left(\mathbf T_{t\rightarrow r}\bar{\mathbf X}_t(\mathbf u)\right)\in\mathcal D,
\ z_{t\rightarrow r}(\mathbf u)>0,
\ \mathbf C_t(\mathbf u)>\tau_c
\right].
\tag{4}
$$

对候选帧 $I_t$ 的每个像素 $\mathbf u$，先将其三维点 $\bar{\mathbf X}_t(\mathbf u)$ 变换到已有关键帧 $I_r$ 的坐标系并投影。投影位于图像范围 $\mathcal D$ 内、深度 $z_{t\rightarrow r}>0$ 且置信度 $\mathbf C_t(\mathbf u)>\tau_c$ 时，该像素计为有效投影。

求和得到 $I_t$ 与关键帧 $I_r$ 的有效重叠数，再对所有 $I_r\in\mathcal K$ 取最大值 $n_t$。最终 reference set 为

$$
\mathcal R
=
\{0\}
\cup
\left\{
t\in\{1,\ldots,N-1\}
\ \middle|\
\frac{n_t}{HW}<0.2
\right\}.
$$

$\mathcal R$ 作为后续 dense matching 的关键帧索引集合。

论文窗口长度为 20，步长为 10；当前实现处理 single window。

**3. Dense matching。** 对关键帧 $I_a \in \mathcal R$，matching head 输出到其余帧 $\mathcal B$ 的 warp 和 confidence：

$$
\left(\mathbf W^{a\rightarrow\mathcal B},\mathbf p^{a\rightarrow\mathcal B}\right)
=
\operatorname{DPT}_{\mathrm{match}}\!\left(
\operatorname{Dec}_{\mathrm{match}}(\mathbf H),a
\right).
\tag{2}
$$

每个关键帧采样 512 个高置信度像素，KITTI 采样 256 个；warp confidence 低于 0.6 的 observation 被移除。

**4. Multi-view tracks 与 observation set。**

$$
\mathcal T_j
=
\left\{
(i,\mathbf u_{ij},\omega_{ij})
\mid i\in\mathcal V_j
\right\},
\qquad
\mathcal O
=
\left\{
(i,j)
\mid j=1,\ldots,J,\ i\in\mathcal V_j
\right\}.
$$

$j$ 是 track 及其 sparse point $\mathbf X_j$ 的编号；$\mathcal V_j$ 是有效观测帧集合；$\mathbf u_{ij}$ 和 $\omega_{ij}$ 是对应的像素坐标与 tracking confidence。$\mathcal O$ 是公式 (5)、(6) 使用的全部 camera-point observations。

**5. Translation averaging。** 固定旋转，联合优化所有 camera centers、sparse points 和 observation depths：

$$
\min_{\{\mathbf c_i\},\{\mathbf X_j\},\{d_{ij}\}}
\sum_{(i,j)\in\mathcal O}
\omega_{ij}\rho\!\left(
\left\|\mathbf X_j-
\left(\mathbf c_i+d_{ij}\mathbf R_i^{\top}\mathbf v_{ij}\right)
\right\|_2^2
\right).
\tag{5}
$$

$\mathcal O$ 包含全部有效 track observations。

**6. Bundle adjustment。** 以 motion averaging 结果初始化，在 $\mathcal O$ 上优化相机位姿、sparse points，以及启用时的内参与畸变：

$$
\min_{\{\mathbf T_i\},\{\mathbf X_j\},\{\mathbf K_i\},\{\boldsymbol\delta_i\}}
\sum_{(i,j)\in\mathcal O}
\omega_{ij}\rho\!\left(
\left\|
\pi\!\left(\mathbf K_i,\boldsymbol\delta_i,\mathbf T_i,\mathbf X_j\right)
-\mathbf u_{ij}
\right\|_2^2
\right).
\tag{6}
$$

BA 优化所有具有有效 observations 的帧；关键帧仅作为 track anchor 和 dense reconstruction 来源。当前实现固定 camera 0 和 point 0；相机标定默认固定。

**7. Dense reconstruction。** 对每个关键帧 $i$，由 BA sparse depth 和 predicted depth 计算 ratios：

$$
z_{ij}^{\mathrm{BA}}
=
[\mathbf T_i^{w2c}\mathbf X_j]_z,
\qquad
r_{ij}
=
\frac{z_{ij}^{\mathrm{BA}}}{D_i^{\mathrm{pred}}(\mathbf u_{ij})}.
$$

RANSAC 估计每个关键帧的尺度并缩放深度：

$$
s_i=\operatorname{RANSAC}(\{r_{ij}\}),
\qquad
D_i^{\mathrm{scaled}}=s_iD_i^{\mathrm{pred}}.
$$

优化后的内参、畸变和位姿用于反投影与融合：

$$
\mathbf X_i^{\mathrm{dense}}(\mathbf u)
=
(\mathbf T_i^{w2c})^{-1}
\pi^{-1}\!\left(
\mathbf K_i,\boldsymbol\delta_i,
\mathbf u,D_i^{\mathrm{scaled}}(\mathbf u)
\right).
$$

## 当前实现的问题：点云分层

- 最终 dense cloud 仅由关键帧生成；四个关键帧对应四组 dense points。
- BA 优化 sparse points 和所有有效相机位姿；dense depths 不参与公式 (6)。
- 每个关键帧独立估计 $s_i$，不同关键帧之间没有尺度一致性约束。
- 单一 $s_i$ 只能校正整体尺度，不能校正局部深度形变和平面倾斜。
- 当前 RANSAC 仅要求至少两个有效 ratios，未设置最小 inlier 数、inlier fraction 和 MAD 上限。
- 当前 track IDs 按关键帧独立创建，未执行 cross-reference track merging。
- 当前 pose graph 显式包含 reference-target edges，未由完整 track 补充 target-target edges。
- 当前使用 Pi3 且 $m=1$，没有 Pi3X metric scale、overlap association 和 loop constraints。
- Sparse cloud 一致而 dense cloud 分层时，误差位于 scale recovery 或 dense back-projection。
- Sparse cloud 在 BA 前已分层时，误差位于 tracks、pose initialization 或 translation averaging。
