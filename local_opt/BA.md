# Glob3R 单窗口优化

```text
Pi3 Eq. (1) geometry
  → Eq. (4) keyframes
  → Eq. (2) dense warps
  → Tracks(rs, Xs_Cr, us, mask, ws)
  → Eq. (5) ray-consistency optimization
  → Eq. (6) sparse bundle adjustment
  → keyframe dense reconstruction
```

`Tracks` 使用 `[S,P]` 组织多视图观测。第 `j` 列始终表示同一条 track：

\[
u_i^j=\texttt{us[i,j]},\qquad
\omega_{ij}=\texttt{ws[i,j]}.
\]

`rs[j]` 是 track 的 keyframe，`Xs_Cr[j]` 是 Pi3 在该 keyframe 坐标系中预测的局部点。共享世界点不通过多帧三角化初始化，而是严格使用 anchor point：

\[
X_j^0=T_{WC,r_j}X_{r_j}(u_{r_j}^j).
\]

其他帧的 Pi3 point map 不会为同一条 track 创建额外三维变量，只提供 Glob3R warp 得到的二维观测。Eq. (5) 优化相机中心、共享点和逐观测 ray depth：

\[
\min_{\{c_i\},\{X_j\},\{d_{ij}\}}
\sum_{(i,j)\in\mathcal O}
\omega_{ij}\rho\!\left(
\left\|X_j-\left(c_i+d_{ij}R_i^\top v_{ij}\right)\right\|_2^2
\right).
\tag{5}
\]

Eq. (6) 继续优化同一组 `X_j` 与相机位姿：

\[
\min_{\{T_i\},\{X_j\}}
\sum_{(i,j)\in\mathcal O}
\omega_{ij}\rho\!\left(
\left\|\pi(K_i,\delta_i,T_i,X_j)-u_i^j\right\|_2^2
\right).
\tag{6}
\]

后端按 DROID BA 的结构拆为 `backend/proj.py`、`backend/chol.py` 和 `backend/ba.py`。Eq. (5) 在线性化时解析消去 $d_{ij}$，Eq. (6) 使用 `lietorch.SE3.retr()` 更新位姿；两者都通过解析 Jacobian、block normal equations、Schur complement 和 Cholesky 求解。状态仍是 Glob3R 的共享 $X_j$，没有恢复 DROID 的逆深度图。当前固定内参与零畸变。
