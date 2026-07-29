# Glob3R 讨论邮件

**主题：Glob3R 多视图匹配嵌入的复现问题与两个相关问题**

尊敬的谭教授，您好：

我叫彭谢昕，来自华中科技大学，目前在华为 2012 实验室黎曼实验室实习。近期我正在结合多视图前馈几何模型改进 MASt3R-Fusion，目前已取得一些积极结果。我也在复现 Glob3R，希望进一步改善前馈重建模型的匹配能力，并为后续 bundle adjustment 提供可靠的 SfM 约束。Glob3R 将前馈几何先验、稠密匹配与全局 SfM 优化相结合的思路对我的工作很有启发。

在复现附录的 Multi-view Match Embedding 模块时，我发现公式 (15)、(16)、(18) 和 (31) 所描述的 reference/target 索引方向，与论文其余部分采用的 reference-to-target warp 方向似乎不一致。我首先严格按照论文公式实现，但该版本无法收敛到有效的 warp 预测。随后，我将相似度、坐标嵌入、DPT 输入及 NLL 监督统一到 reference 网格，并预测对应的 target 位置；在相同训练设置下，修改后的版本可以正常训练。因此，这并非单纯的符号理解疑问，而是我在实际训练中观察到的稳定差异。具体推导、修改后的公式和实验现象已整理在附件中，想请您确认附件中的方向是否才是作者实际实现所采用的定义。

此外，还有两个相关问题想向您请教：

1. Glob3R 将每个窗口的第一帧初始化为 reference keyframe，而后续关键帧由公式 (4) 决定。由于第一帧由窗口切分位置决定，未必与其余视图具有最佳重叠，请问您是否评估过初始 reference 的选择对匹配质量和 track construction 的影响，或尝试过其他 reference 选择策略？

2. Glob3R 通过 reference-anchored tracks 和重叠窗口建立全局关联。在大视角变化、非连续运动、低帧率采样或严重遮挡等复杂运动场景中，单个 reference 与其他视图的共视关系可能迅速减弱。请问系统如何维持可靠且全局一致的多视图关联？其鲁棒性主要来自关键帧选择和窗口重叠，还是还采用了跨 reference 的关联、一致性检查或其他全局机制？

感谢您和团队的优秀工作，也感谢您抽出时间阅读。如果有帮助，我很愿意进一步分享实现细节、训练曲线和其他复现实验结果。

此致  
敬礼！

彭谢昕  
华中科技大学  
华为 2012 实验室黎曼实验室实习生  
p-xiexin@outlook.com

---

## 附件：Multi-view Match Embedding 中的方向问题

### 1. 论文原文表述及对应的训练现象

论文在公式 (28) 中将 ground-truth warp 定义为从 reference 图像 $a$ 到 target 图像 $b$：

$$
W^{*\,a\rightarrow b}(x^a)=x^{a\rightarrow b}.
\tag{28}
$$

这意味着 warp 应以 reference patch $n$ 为索引，并输出其对应的 target 位置 $m_n^*$。

但在 Multi-view Match Embedding 中，公式 (12) 将相似度写为：

$$
S_{mn}^{a\rightarrow b}
=
\exp\left(
\frac{1}{\tau}\operatorname{cosim}(z_m^a,z_n^b)
\right).
\tag{12}
$$

随后，公式 (15)、(16) 编码 reference 坐标，并为索引为 $m$ 的 patch 聚合这些坐标：

$$
\chi_n^a=\gamma(p_n^a).
\tag{15}
$$

$$
\chi_m^{a\rightarrow b}=\sum_n S_{mn}^{a\rightarrow b}\chi_n^a.
\tag{16}
$$

公式 (18) 又将该 embedding 与 target token 拼接：

$$
F^{a\rightarrow b}
=
\operatorname{Proj}\left(Z^b\oplus\chi^{a\rightarrow b}\right).
\tag{18}
$$

公式 (31) 同样将相似度矩阵的每一行解释为一个 target patch $m$，并监督其对应的 reference patch $n_m^*$：

$$
\mathcal{L}_{\mathrm{NLL}}^{a\rightarrow b}
=
-\frac{1}{|\Omega_{\mathrm{patch}}^{a\rightarrow b}|}
\sum_{m\in\Omega_{\mathrm{patch}}^{a\rightarrow b}}
\log\left(
\operatorname{Softmax}(S_{m:}^{a\rightarrow b})_{n_m^*}
\right).
\tag{31}
$$

因此，按照公式 (16)、(18) 和 (31)，行索引 $m$ 表示 target patch，列索引 $n$ 表示 reference patch，整个计算链实际描述的是 target-to-reference 映射。这不仅与公式 (28) 的 reference-to-target warp 相反，也与 refinement 中在 reference 网格上采样 target features 的方向不一致。此外，公式 (12) 已对余弦相似度取指数，而公式 (31) 又对 $S$ 执行 Softmax，按字面实现还会产生二次指数化。

我首先严格按照上述方向实现并训练了网络。训练过程中，NLL loss 能够下降并表现出一定的收敛趋势，但 warp loss 和 confidence loss 始终明显震荡，网络最终无法产生有效的 warp 预测。这说明相似度分支可以学习到部分 patch 分类关系，但由 match embedding 到 DPT 和 refinement 的后续回归过程没有形成一致的空间对应。

### 2. 修改后的形式及训练结果

为使整个计算链与 reference-to-target warp 一致，我将相似度矩阵定义为 reference 行、target 列，并沿 target 维进行归一化：

$$
L_{nm}^{a\rightarrow b}
=
\frac{1}{\tau}\operatorname{cosim}(z_n^a,z_m^b),
\qquad
P_{nm}^{a\rightarrow b}
=
\frac{\exp(L_{nm}^{a\rightarrow b})}
{\sum_{m'}\exp(L_{nm'}^{a\rightarrow b})}.
\tag{12}
$$

随后编码 target 坐标，并将其聚合回每个 reference patch：

$$
\chi_m^b=\gamma(p_m^b).
\tag{15}
$$

$$
\chi_n^{a\rightarrow b}
=
\sum_m P_{nm}^{a\rightarrow b}\chi_m^b.
\tag{16}
$$

DPT 的输入也保持在 reference 网格：

$$
F^{a\rightarrow b}
=
\operatorname{Proj}\left(Z^a\oplus\chi^{a\rightarrow b}\right).
\tag{18}
$$

对于每个有效的 reference patch $n$，NLL 监督其对应的 target patch $m_n^*$：

$$
\mathcal{L}_{\mathrm{NLL}}^{a\rightarrow b}
=
-\frac{1}{|\Omega_{\mathrm{valid}}|}
\sum_{n\in\Omega_{\mathrm{valid}}}
\log P_{n,m_n^*}^{a\rightarrow b}.
\tag{31}
$$

实现中直接将 $L$ 作为 cross-entropy logits，并用 $P=\operatorname{Softmax}(L)$ 进行坐标聚合。这样可以避免先对余弦相似度取指数、再在 NLL 中重复执行 Softmax 所造成的二次指数化。

在相同的数据、初始化和训练设置下，将 match embedding、DPT 输入和 NLL 统一为上述 reference-grid 定义后，网络可以正常训练，并能够学习到有效的 warp 预测。对照结果表明，修改后的方向与论文定义的 reference-to-target warp 以及后续 refinement 过程是一致的。
