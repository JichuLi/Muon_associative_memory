# PTB Mechanism Study v2 中文说明

生成时间：Unix time `1788405553`。

这个目录是用于上传 GitHub 的精简发布包。原始工作目录中的 legacy 输出、WikiText 文件、临时缓存、日志和控制文件没有放进来；这里保留 PTB v2 实验所需的核心代码、canonical 配置、PTB 预处理数据、结构化结果记录、全部 v2 figures，以及这份中文 README。

本 README 重点解释实验 setup、H0 inverse GD 的公式、目录结构、每类图片的 metric，以及 30 个 PTB setting 的数据生成方式。

## 1. 实验 Setup

### 1.1 基本对象

- 只做 PTB，词表大小 `n=5000`；不包含 WikiText。
- 表示维度 `d=256`。
- 每个实验只跑一个 seed：`target_seed=0`，`rep_seed=0`。
- 每条训练曲线跑 `200` 个 full-batch update。
- 所有 setting 都是 fixed zero-bias：输出 bias 固定为 0，不学习 bias，不使用 unigram bias。
- representation 使用随机 normalized input/output features：输出侧 `U in R^{n x d}`，输入侧 `E in R^{d x n}`，每个输出向量和每个 context 向量都做单位范数归一化。
- loss 计算用 float64 accumulation；参数/主要矩阵用 float32；不使用 momentum，不使用 weight decay。

记 context 分布为 $\pi_x$，目标条件分布为 $P(y|x)$。模型参数为 $W\in\mathbb{R}^{d\times d}$。因为 zero-bias，logit 是

$$
z_{yx}(W)=u_y^\top W e_x.
$$

模型条件分布为

$$
q_W(y|x)=\frac{\exp(z_{yx}(W))}{\sum_{y'}\exp(z_{y'x}(W))}.
$$

优化目标是 full-batch cross entropy

$$
L(W)=\sum_x \pi_x\left[\log\sum_y \exp(z_{yx}(W))-\sum_y P(y|x)z_{yx}(W)\right].
$$

令 residual 矩阵

$$
R_{yx}=\pi_x(q_W(y|x)-P(y|x)).
$$

则梯度为

$$
G(W)=\nabla_W L(W)=U^\top R E^\top.
$$

### 1.2 优化器

每个 setting 都跑同一组优化器：`gd_ls`, `muon_ls`, `h0_ls`, `signgd_ls`, `gd_const`, `ngd_const`, `muon_const`, `h0_const`, `signgd_const`。

- `gd`: $D=-G$。
- `ngd`: $D=-G/\lVert G\rVert_F$。
- `muon`: 对梯度矩阵做 polar/SVD 方向，$G=\tilde U\Sigma \tilde V^\top$，保留 $\sigma_i>10^{-7}\sigma_1$ 的方向，$D=-\tilde U_r\tilde V_r^\top$。
- `signgd`: $D=-\operatorname{sign}(G)$，逐 entry 取符号。
- `h0`: frozen initial-Hessian inverse direction，公式见下一节。

后缀含义：

- `_ls`：每一步沿当前方向做 exact numerical ray line-search，直接在该方向上最小化真实 CE。
- `_const`：使用固定 learning rate；每个 optimizer、每个 criterion 独立选择自己的 best constant LR。

### 1.3 H0 inverse GD 的推导

H0 inverse GD 不是用训练出来的 W*，也不是用 Muon/GD 的结果。它是在初始化点 $W=0$ 处，把 softmax CE 的 Hessian 固定下来，然后每一步用这个固定 Hessian 的 inverse 预条件梯度。

先写二阶变化。对一个扰动 $\Delta$，在一般初始 bias $b_0$ 下有初始分布 $q_0=\operatorname{softmax}(b_0)$。本实验 fixed zero-bias，所以 $q_0(y)=1/n$。在 $W=0$ 处，

$$
\delta^2 L(0)[\Delta,\Delta]
=\sum_x \pi_x\,\operatorname{Var}_{y\sim q_0}\left(u_y^\top \Delta e_x\right).
$$

定义输出侧初始协方差

$$
A_0=\sum_y q_0(y)(u_y-\bar u)(u_y-\bar u)^\top,\qquad
\bar u=\sum_y q_0(y)u_y,
$$

以及输入/context 侧协方差

$$
C_0=\sum_x \pi_x e_x e_x^\top.
$$

则二阶型可以写成

$$
\delta^2 L(0)[\Delta,\Delta]=\operatorname{tr}(\Delta^\top A_0\Delta C_0).
$$

对应的 Hessian 线性算子是

$$
H_0[\Delta]=A_0\Delta C_0,
$$

也就是 vectorize 后的 Kronecker 形式

$$
\operatorname{vec}(H_0[\Delta])=(C_0^\top\otimes A_0)\operatorname{vec}(\Delta).
$$

Newton 型方向要解 $H_0[D]=-G$，所以

$$
D_{\mathrm{H0}}=-A_0^{-1}GC_0^{-1}.
$$

代码里对应 `core.py` 的 `self.Ainv @ G @ self.Cinv`。这里没有加 hidden ridge；如果初始协方差不是正定，代码会直接报错，而不是偷偷正则化。`h0_ls` 在这个方向上做 line-search；`h0_const` 使用固定 LR 并通过 criterion 选择。

直观上，H0 inverse GD 用的是初始化处 softmax 曲率的 separable/Kronecker 结构：左边校正 output-feature 几何，右边校正 context-feature 几何。它是非常强的 inverse-geometry baseline，所以最后总结 Muon benefit 时会单独给出“排除 H0 inverse GD 后”的比较。

### 1.4 两套 best 选择标准

每个 constant-lr optimizer 都有两套 best LR 选择；line-search optimizer 没有 LR sweep。

- `auc`：legacy 标准，直接使用每条 run metadata 里的 `auc`，定义为 steps 0-200 上的 mean log CE。lower is better。这个标准保持和旧实验一致，不重新解释、不重新发明。
- `late`：新标准，使用 steps 120-150 上 raw CE 的平均值。lower is better。它更接近“只关心后期/final loss”的真实训练读法。

注意：loss 图画的是 raw CE versus step。为了避免某条发散曲线把其他曲线压扁，图的 y-axis 使用 log scale；这不是在画 log-loss metric，也不是 relative gap。

### 1.5 W* reference

W* 不由 H0、Muon、GD、NGD、SignGD 中任何一个参与比较的 optimizer 定义。当前 v2 使用独立的 `torch.optim.LBFGS` 加 `strong_wolfe` line search，从 zero-bias 初始化尝试求 reference。

- `hard_copy` 被标记为 `skipped_definitely_no_finite_optimum`，因为 deterministic hard-copy 在 zero-bias softmax 下没有 finite minimizer。
- 其他 setting 都尝试 L-BFGS；当前状态是 conservative 的 `approximate` 或 `attempted_zero_support_approximate`。
- 只要独立 L-BFGS W* attempt 有 `reference_loss`，loss 图就画水平 `L(W*)` ref line；legend 会标出它是 `solved` 还是 approximate。`hard_copy` 这种没有 finite W* 的 setting 不画这条线。
- `wstar.json` 记录的 W* properties 包括：reference loss、gradient Frobenius norm、`W` 的 Frobenius/operator/nuclear norm、stable rank、完整奇异值谱、与 `B` 的 cosine alignment、与 `H0^{-1}B` 的 cosine alignment、prediction marginal TV、prediction entropy、以及 head/middle/tail bucket CE。

## 2. 主目录结构

- `README.md`：当前中文主说明文件，适合作为 GitHub repo 首页。
- `config/study_v2.json`：v2 实验的唯一 canonical 配置，包含 30 个 setting、optimizer 列表、LR grid、selection criteria、W* policy。
- `data/ptb_v5000.pt`：PTB 预处理后的 5000-vocab 语料数据，用于复现目标分布和训练权重。
- `data/README.md`：说明数据文件的用途。
- `cache/ptb/samples_0.npz`：fixed sampled target 使用的 PTB 条件采样缓存。
- `cache/ptb/groups_k32_random0_seed0.npz`：`prototype_32` 使用的 32 组 context prototype 缓存。
- `results/study_v2.sqlite`：主结果数据库，保存 run traces、diagnostics、case status、selection metadata。
- `results/v2/<case_id>/`：每个 setting 的结构化结果目录。
- `figures/v2/<case_id>/`：每个 setting 的图片目录。
- `results/analysis_v2/summary_by_case.csv`：每个 setting、每个 criterion 的跨 optimizer 摘要。
- `results/analysis_v2/selected_methods.csv`：每个 setting、每个 criterion、每个 optimizer 被选中曲线的指标。
- `results/overview_v2.csv` / `results/inventory_v2.csv` / `results/inventory_v2.json` / `results/validation_v2.json`：实验总览、预计耗时记录和完整性验证记录。
- `targets.py`：所有 $P(y|x)$ 数据生成/intervention 的实现。
- `core.py`：loss、gradient、H0 inverse、Muon、SignGD、line-search、diagnostics 的核心实现。
- `run_case_v2.py` / `run_study_v2.py`：单 setting 和整组实验的运行入口。
- `plots_v2.py`：每个 setting 的图片生成逻辑。
- `wstar.py`：独立 L-BFGS W* reference attempt。
- `analyze_v2.py`：生成 `results/analysis_v2` 的汇总表。
- `validate_v2.py`：完整性验证脚本。
- `write_readme_v2.py` / `write_readme_v2_cn.py`：README 生成脚本；GitHub 首页使用当前 `README.md`。

## 3. 单个 Setting 文件夹里有什么

下面的结构对所有 setting 基本一致。设 setting 名为 `<case_id>`。

### 3.1 `results/v2/<case_id>/metadata/`

- `case_specification.json`：该 setting 的完整配置快照，包括 target recipe、zero-bias、methods、seed、protocol signature。
- `target_seed_0.json`：目标分布统计，例如 conditional entropy、weighted top-1、mutual information、min probability、target/output marginal TV 等。
- `representation_target0_rep0.json`：随机 representation 的 hash、初始几何 condition number 等。
- `figure_manifest.json`：该 setting 计划/生成的图片清单。

### 3.2 `results/v2/<case_id>/selection/`

- `selection.json`：legacy AUC 标准下，每个 optimizer 的 best LR。结构保持旧风格：不是全局选一个 winner，而是每个 method 自己选自己的 LR。
- `selection_late.json`：late 标准下，每个 optimizer 的 best LR。
- `selected.csv`：AUC-selected 曲线的 run metadata。
- `selected_late.csv`：late-selected 曲线的 run metadata。

### 3.3 `results/v2/<case_id>/minimizer/`

- `wstar.json`：独立 L-BFGS reference attempt 的完整记录和 W* properties。`reference_method` 应为 `lbfgs`；不会是 H0/Muon/GD。
- `wstar_properties.csv`：把 `wstar.json` 中的 scalar property 摘出来，方便做表格分析。
- `wstar_progress.json`：L-BFGS reference attempt 的进度/trace。

### 3.4 `figures/v2/<case_id>/loss/`

- `loss_vs_step_auc_selected.png`：AUC 标准选出的每个 optimizer 曲线。x-axis 是 update step，y-axis 是 raw training CE in nats；y-axis 用 log scale 只是为了可读性。每条线对应一个 optimizer 的 selected LR 或 line-search 曲线。
- `loss_vs_step_late_selected.png`：late 标准选出的每个 optimizer 曲线。画法同上，但 constant LR 来自 `selection_late.json`。
- 水平 minimum/ref line：只要 `wstar.json` 中有 `reference_loss` 就画 `L(W*)` ref line；legend 中会写出 W* status。`hard_copy` 没有 finite W*，因此没有这条线。

### 3.5 `figures/v2/<case_id>/selection/`

- `lr_sweeps_auc.png`：constant-lr sweep 图。每个 constant optimizer 一个小面板；x-axis 是 LR，log scale；y-axis 是 AUC score，即 mean log CE。黑色星号标出该 optimizer 在 AUC 标准下选中的 LR。
- `lr_sweeps_late.png`：同样的 LR sweep，但 y-axis 是 steps 120-150 的 mean raw CE。黑色星号标出 late 标准选中的 LR。

### 3.6 `figures/v2/<case_id>/dynamics/`

- `optimizer_dynamics_auc_selected.png` / `optimizer_dynamics_late_selected.png`：六个动态指标面板。
  - `eta`：optimizer 原生 update coefficient；line-search 是 ray 上选出的系数，constant 是固定 LR。
  - `eta_fro`：实际 update 的 Frobenius 长度，也就是沿方向走的 raw ray length。
  - `grad_fro`：梯度 Frobenius norm $\lVert G\rVert_F$。
  - `prediction_entropy`：模型预测分布的条件熵 $\sum_x\pi_x H(q_W(\cdot|x))$。
  - `prediction_marginal_tv`：模型预测边际 $q_W\pi$ 与目标输出边际 $p$ 的 total variation distance。
  - `grad_top32_energy`：梯度前 32 个奇异值能量占比 $\sum_{i\le 32}\sigma_i^2/\sum_i\sigma_i^2$。
- `frequency_ce_auc_selected.png` / `frequency_ce_late_selected.png`：按 context frequency 分成 head/middle/tail 三个 bucket 后，各 bucket 内 CE 随 step 的变化。
- `geometry_auc_selected.png` / `geometry_late_selected.png`：曲率和梯度几何面板，包括当前方向曲率、当前/初始曲率比、gradient nuclear effective rank、marginal-gradient norm ratio。
- `gradient_spectra_auc_selected.png` / `gradient_spectra_late_selected.png`：在 diagnostic steps `0,20,100,120,150,200` 上画梯度奇异值谱，纵轴是 $\sigma_i/\sigma_1$，横轴是奇异值 index；虚线是 Muon polar cutoff `1e-7`。
- `heldout_auc_selected.png` / `heldout_late_selected.png`：可选 held-out CE 图，目前只在 `ptb` 和 `permuted_entries` 中出现；不用于 LR 选择。
- `same_state` probes 在 v2 中不画。

### 3.7 `figures/v2/<case_id>/minimizer/`

- `wstar_scalar_properties.png`：L-BFGS reference candidate 的 scalar summary。它把 `wstar.json` 里已经记录的 reference loss、gradient norm、矩阵范数、stable rank、nuclear effective rank、相对阈值 rank、alignment、prediction entropy、prediction marginal TV、head/middle/tail CE 汇总到一张图里。
- `wstar_singular_values.png`：L-BFGS reference candidate 的 W 矩阵奇异值，以及归一化奇异值谱。图中红色竖向虚线标出 stable rank，定义为 $\lVert W\rVert_F^2/\lVert W\rVert_{op}^2$。
- `wstar_bucket_ce.png`：reference candidate 在 head/middle/tail context bucket 上的 CE。
- `wstar_solver_traces.png`：L-BFGS reference attempt 的 loss 和 gradient Frobenius norm 随 reference step 的变化。

## 4. 每个 Setting 的含义和数据生成方式

通用记号：$Q_0(y|x)$ 是未平滑 PTB bigram 条件分布，$Q_{0.001}(y|x)=(1-0.001)Q_0(y|x)+0.001p(y)$，$p(y)$ 是输出 unigram，$\pi_x$ 是 context unigram。除非特别说明，每个 setting 最后都会按列重新 normalize 成合法条件分布。

### `ptb`

- family: `natural_conditional`
- target recipe: `{"kind": "ptb", "rho": 0.001}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是自然 PTB bigram 条件分布，是整组实验的主参照。
- 令原始经验 bigram 条件分布为 $Q(y|x)$，输出 unigram 为 $p(y)$。本 setting 使用平滑后的 $P(y|x)=(1-\rho)Q(y|x)+\rho p(y)$，其中 $\rho=0.001$。
- 这个 setting 仍然保留已有平滑版本，以最大化复用 legacy 数据；它不是新的 rho sweep。

### `sampled_onehot`

- family: `finite_support`
- target recipe: `{"kind": "sample_k", "k": 1}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 每个 context 固定采样一个目标 token，得到确定性 one-hot 目标。
- 对每个 $x$，从 $Q(\cdot|x)$ 采样 $y_x$，然后设 $P(y|x)=\mathbf{1}[y=y_x]$。seed=0 时第一列样本沿用已有 legacy fixed-onehot 文件。
- 它保留 PTB 的 context-dependent 采样来源，但训练目标本身是有限支持/可分倾向的。

### `hard_copy`

- family: `copy_am`
- target recipe: `{"kind": "copy", "association": 1}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是最硬的 copy associative-memory 目标。
- 输入输出共享同一个 5000 词表索引，目标为 $P(y|x)=\mathbf{1}[y=x]$。
- 在 zero-bias softmax 下，这是确定性分类目标，没有有限范数的 softmax minimizer；因此 W* 明确跳过。

### `soft_copy_05`

- family: `copy_am`
- target recipe: `{"kind": "copy", "association": 0.5}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 50% copy 加 50% unigram 背景的 soft copy。
- 令 $a=0.5$，代码中的 copy 背景使用 context unigram $\pi$ 作为输出侧背景：$P(y|x)=a\mathbf{1}[y=x]+(1-a)\pi_y$。
- 它测试对角 copy 结构在非确定性背景下是否仍能放大某些优化器的优势。

### `rewired`

- family: `null_no_association`
- target recipe: `{"kind": "rewire", "mix": 1}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 marginal-preserving rewire，是去掉真实 context-target association 的 null control。
- 实现上把训练 bigram 事件中的输出 token 全局随机重排，同时保持每个 context 的出现次数和每个 output 的出现次数不变。
- 记重排后的条件分布为 $R(y|x)$，本 setting 使用 $P(y|x)=0.999R(y|x)+0.001p(y)$。

### `independent`

- family: `null_no_association`
- target recipe: `{"kind": "association", "alpha": 0}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是完全 independent target。
- 每一列都等于输出 unigram：$P(y|x)=p(y)$。也就是 $Y$ 与 $X$ 独立，没有任何 bigram association。
- 它和 rewire 都是“去 association”的 control，但构造不同：independent 的每列完全相同；rewire 仍保留随机化后的列形状和边际计数。

### `flat_context`

- family: `context_frequency`
- target recipe: `{"kind": "context_frequency", "gamma": 0}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 目标条件分布仍是 PTB default，但 context 权重被改成 uniform。
- 条件分布 $P(y|x)=Q_{0.001}(y|x)$，context 权重改为 $\pi'_x\propto \pi_x^0$，所以所有 context 权重相同。
- 它隔离了 context heavy-tail 权重本身对优化轨迹的影响。

### `permuted_entries`

- family: `matrix_geometry`
- target recipe: `{"kind": "ptb", "rho": 0.001}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这个 setting 不改变 $P(y|x)$，只破坏 Muon 看到的矩阵几何。
- 目标仍是 $P(y|x)=Q_{0.001}(y|x)$。但对 Muon，在计算 $\operatorname{polar}(G)$ 前，把 $d\times d$ 梯度矩阵的 entry 做固定随机置换；取完 polar 后再映射回物理坐标。
- GD、NGD、H0 inverse GD、SignGD 仍然使用原始物理梯度；因此它专门测试 Muon 是否依赖有意义的矩阵排列，而不只是 entrywise heavy-tail。

### `onehot_mix_0.5`

- family: `finite_support`
- target recipe: `{"kind": "onehot_mix", "alpha": 0.5}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 one-hot sampled target 与真实经验 bigram 的 50/50 混合。
- 令 $S_1(y|x)$ 是 sampled-onehot 目标，$Q_0(y|x)$ 是未平滑 PTB 条件分布。本 setting 使用 $P=(1-\alpha)S_1+\alpha Q_0$，$\alpha=0.5$。
- 这里 $\alpha$ 表示 empirical association 的比例。

### `association_0.1`

- family: `association_strength`
- target recipe: `{"kind": "association", "alpha": 0.1}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 10% empirical association 加 90% independent background。
- 使用 $P(y|x)=\alpha Q_0(y|x)+(1-\alpha)p(y)$，其中 $\alpha=0.1$。
- 它测试很弱的真实 bigram association 是否足以改变优化器排序。

### `rewire_mix_0.5`

- family: `null_no_association`
- target recipe: `{"kind": "rewire", "mix": 0.5}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是原始 PTB 和 marginal-preserving rewire 的 50/50 混合。
- 令 $Q_{0.001}$ 是平滑 PTB，$R_{0.001}=0.999R+0.001p$ 是平滑 rewired target。本 setting 使用 $P=0.5Q_{0.001}+0.5R_{0.001}$。
- 它保留一半真实 association，同时混入一半保持边际但打乱 association 的结构。

### `context_gamma0.5`

- family: `context_frequency`
- target recipe: `{"kind": "context_frequency", "gamma": 0.5}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 条件分布仍是 PTB default，但 context 权重按 sqrt frequency 重新加权。
- 目标 $P=Q_{0.001}$，权重 $\pi'_x\propto \pi_x^{0.5}$。
- 这会压低头部 context 的相对权重，让 context 分布更平。

### `onehot_mix_0.25`

- family: `finite_support`
- target recipe: `{"kind": "onehot_mix", "alpha": 0.25}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 one-hot sampled target 与经验 bigram 的混合，其中 empirical 比例为 25%。
- $P=0.75S_1+0.25Q_0$。
- 它比 50% empirical 更接近 sparse one-hot 目标。

### `onehot_mix_0.75`

- family: `finite_support`
- target recipe: `{"kind": "onehot_mix", "alpha": 0.75}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 one-hot sampled target 与经验 bigram 的混合，其中 empirical 比例为 75%。
- $P=0.25S_1+0.75Q_0$。
- 它比 50% empirical 更接近真实 PTB 条件分布。

### `sample_k4`

- family: `finite_support`
- target recipe: `{"kind": "sample_k", "k": 4}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 每个 context 固定采样 4 个目标样本，形成稀疏 soft target。
- 对每个 $x$，从 $Q(\cdot|x)$ 采样 $k=4$ 次，设每次样本质量为 $1/k$。若同一个 token 被重复采到，其概率质量会累加。
- 它位于 deterministic one-hot 和 dense empirical distribution 之间。

### `sample_k16`

- family: `finite_support`
- target recipe: `{"kind": "sample_k", "k": 16}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 每个 context 固定采样 16 个目标样本，形成较不稀疏的 fixed-sample target。
- 构造同 sample_k4，但 $k=16$。
- 它测试随着每列支持集变大，Muon 相对 scalar baseline 的优势是否减弱。

### `association_0.5`

- family: `association_strength`
- target recipe: `{"kind": "association", "alpha": 0.5}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 50% empirical association 加 50% independent background。
- $P(y|x)=0.5Q_0(y|x)+0.5p(y)$。
- 它比 association_0.1 有更强的真实 bigram association。

### `context_gamma2`

- family: `context_frequency`
- target recipe: `{"kind": "context_frequency", "gamma": 2}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 条件分布仍是 PTB default，但 context 权重按 frequency squared 重新加权。
- 目标 $P=Q_{0.001}$，权重 $\pi'_x\propto \pi_x^2$。
- 这会显著放大头部 context 的训练权重。

### `copy_assoc_0.1`

- family: `copy_am`
- target recipe: `{"kind": "copy", "association": 0.1}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 10% hard copy 加 90% independent background。
- 令 $a=0.1$，$P(y|x)=a\mathbf{1}[y=x]+(1-a)\pi_y$。
- 它测试很弱的 diagonal copy component 是否足以改变优化器优势。

### `prototype_32`

- family: `shared_structure`
- target recipe: `{"kind": "prototype", "k": 32, "residual": 0}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 shared conditional prototypes setting。
- 先用数据侧 Hellinger geometry 对 context 分成 32 组；每组的 prototype 是组内 PTB 条件列的 $\pi$-加权平均。
- 若 $g(x)$ 是 context 的组，$\bar P_g(y)=\sum_{x:g(x)=g}\pi_xQ_{0.001}(y|x)/\sum_{x:g(x)=g}\pi_x$，则 $P(y|x)=\bar P_{g(x)}(y)$。residual=0，所以组内所有 context 条件分布完全相同。

### `argmax_ptb`

- family: `finite_support_new`
- target recipe: `{"kind": "argmax"}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 每个 context 只保留 PTB 条件分布中概率最大的 output。
- 令 $y^*(x)=\arg\max_y Q(y|x)$，目标为 $P(y|x)=\mathbf{1}[y=y^*(x)]$。
- 它是 PTB association 的 deterministic argmax 版本。

### `top2_renormalized`

- family: `finite_support_new`
- target recipe: `{"kind": "top_k", "k": 2, "background": 0.0}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 每个 context 只保留 PTB top-2 outputs 并重新归一化。
- 从未平滑 $Q_0(\cdot|x)$ 中取概率最大的 2 个 token，其他位置置零，再把该列归一化。background=0。
- 这是 zero-support sparse target，因此 W* 只能做 approximate reference attempt。

### `top8_renormalized`

- family: `finite_support_new`
- target recipe: `{"kind": "top_k", "k": 8, "background": 0.0}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 每个 context 只保留 PTB top-8 outputs 并重新归一化。
- 构造同 top2，但 $k=8$，background=0。
- 它比 top2 支持集更大，但仍是 sparse/zero-support target。

### `frequency_matched_random_onehot`

- family: `random_control_new`
- target recipe: `{"kind": "frequency_matched_onehot"}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 frequency-matched random one-hot control。
- 代码按 $-0.5(\pi+p)$ 的频率排序，把 token 分成很小的相邻频率 bin，并在 bin 内随机置换 label。
- 然后令 $P(y|x)=\mathbf{1}[y=\tilde y_x]$。它保留“context 频率与 label 频率大致匹配”的 heavy-tail 结构，但不使用真实 bigram association。

### `frequency_matched_random_soft_k4`

- family: `random_control_new`
- target recipe: `{"kind": "frequency_matched_soft_k", "k": 4, "bin_size": 8}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 frequency-matched random soft-k control。
- 构造方式类似 frequency_matched_random_onehot，但每个 context 有 $k=4$ 个随机 label，bin_size=8。
- 每个 label 获得 $1/4$ 概率质量，重复 label 的质量累加。

### `random_context_column_permutation`

- family: `alignment_new`
- target recipe: `{"kind": "context_column_permutation"}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是随机 context-column permutation。
- 令 $\sigma$ 是 context 列的固定随机置换，设 $P(\cdot|x)=Q_{0.001}(\cdot|\sigma(x))$。
- 它保留 PTB 条件列的多样性和形状，但切断这些列与原 context identity/frequency 的对应关系。

### `anti_frequency_entropy_alignment`

- family: `alignment_new`
- target recipe: `{"kind": "entropy_context_alignment", "mode": "high_frequency_low_entropy"}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 entropy-context alignment control。
- 先计算 PTB 条件列的熵 $H(Q(\cdot|x))$，再把低熵条件列分配给最高频 context。
- 具体地，context slot 按 $\pi_x$ 从大到小排序，条件列按熵从小到大排序，然后逐一配对。
- 它制造“高频 context 对应低熵 target”的结构，用来测试频率与条件确定性的耦合。

### `head_background_soft_target`

- family: `background_new`
- target recipe: `{"kind": "background_mix", "bucket": "head", "count": 1000, "background": 0.3}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 head-output background soft target。
- 先取输出 unigram $p$ 最大的前 1000 个 token，并把 $p$ 限制在这些 token 上重新归一化为 $b_{	ext{head}}$。
- 使用 $P(y|x)=0.7Q_{0.001}(y|x)+0.3b_{\text{head}}(y)$。

### `tail_background_soft_target`

- family: `background_new`
- target recipe: `{"kind": "background_mix", "bucket": "tail", "count": 2000, "background": 0.3}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 tail-output background soft target。
- 先取输出 unigram $p$ 最小的 2000 个 token，并把 $p$ 限制在这些 token 上重新归一化为 $b_{	ext{tail}}$。
- 使用 $P(y|x)=0.7Q_{0.001}(y|x)+0.3b_{\text{tail}}(y)$。

### `class_block_soft_target`

- family: `shared_structure_new`
- target recipe: `{"kind": "class_block_soft", "blocks": 32, "background": 0.05}`
- 共同设置：zero-bias，target_seed=0，rep_seed=0，200 update steps，9 个 optimizer 全部运行。
- 这是 frequency-block class target。
- 先按输出 unigram $p$ 把 5000 个 output 划成 32 个近似等 $p$-mass 的频率 block。对每个 context，找到其 PTB argmax output 所在 block。
- 主目标是在该 block 内按 $p$ 归一化的分布；最后混入 5% 全局 $p$：$P(y|x)=0.95b_{g(y^*(x))}(y)+0.05p(y)$。
- 它显式构造共享 class/block 结构，同时保持 full support。

## 5. 大概总结：Muon 在哪里明显好

先说明比较口径。这里的“显著”不是统计显著性，因为本轮只跑了一个 seed；它指 effect-size 明显。定义

$$
\Delta_{\text{non-H0}}=\text{score(best non-H0 non-Muon optimizer)}-\text{score(best Muon optimizer)}.
$$

lower score is better，所以 $\Delta_{\text{non-H0}}>0$ 表示 Muon 比除了 H0 inverse GD 以外的所有优化器都好。non-H0 non-Muon optimizer 包括 GD、NGD、SignGD 的 line-search/constant 版本。下面采用保守阈值：AUC 下 $\Delta_{\text{non-H0}}\ge 0.02$，late 下 $\Delta_{\text{non-H0}}\ge 0.05$ nats。

### 5.1 AUC criterion

- 全部 optimizer 的 best method 计数：`h0_ls` 27，`h0_const` 2，`muon_ls` 1。
- 排除 H0 inverse GD 后，Muon 优于 best non-H0 baseline 的 setting 数：24/30。
- 不排除 H0 时，Muon 优于 best non-Muon 的 setting 数：1/30。
- H0 inverse GD 优于 Muon 的 setting 数：29/30。

按上述阈值，Muon 明显好于 non-H0 baselines 的 setting 是：

- `frequency_matched_random_soft_k4`：Δ=+0.377798；best Muon `muon_ls`；best non-H0 baseline `signgd_const`；全 optimizer best `h0_ls`。
- `sample_k4`：Δ=+0.249013；best Muon `muon_ls`；best non-H0 baseline `signgd_const`；全 optimizer best `h0_ls`。
- `top2_renormalized`：Δ=+0.211829；best Muon `muon_const`；best non-H0 baseline `gd_const`；全 optimizer best `h0_ls`。
- `onehot_mix_0.25`：Δ=+0.176114；best Muon `muon_ls`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `top8_renormalized`：Δ=+0.127901；best Muon `muon_ls`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `onehot_mix_0.5`：Δ=+0.118098；best Muon `muon_ls`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `sample_k16`：Δ=+0.10844；best Muon `muon_ls`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `soft_copy_05`：Δ=+0.102702；best Muon `muon_ls`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `random_context_column_permutation`：Δ=+0.0989313；best Muon `muon_ls`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `sampled_onehot`：Δ=+0.0740187；best Muon `muon_ls`；best non-H0 baseline `gd_ls`；全 optimizer best `muon_ls`。
- `onehot_mix_0.75`：Δ=+0.0539968；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `class_block_soft_target`：Δ=+0.0288349；best Muon `muon_ls`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `argmax_ptb`：Δ=+0.0249253；best Muon `muon_ls`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `ptb`：Δ=+0.0244042；best Muon `muon_ls`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。

AUC 下的核心读法：Muon 相对普通 scalar/no-H0 方法的优势主要出现在 sparse/有限支持目标和一些结构化目标上，例如 fixed samples、top-k、onehot/empirical mixture、soft copy、PTB default。但如果把 H0 inverse GD 也放进比较，H0 通常更强。

### 5.2 Late criterion

- 全部 optimizer 的 best method 计数：`h0_ls` 24，`h0_const` 3，`muon_ls` 2，`gd_ls` 1。
- 排除 H0 inverse GD 后，Muon 优于 best non-H0 baseline 的 setting 数：24/30。
- 不排除 H0 时，Muon 优于 best non-Muon 的 setting 数：2/30。
- H0 inverse GD 优于 Muon 的 setting 数：28/30。

按上述阈值，Muon 明显好于 non-H0 baselines 的 setting 是：

- `frequency_matched_random_soft_k4`：Δ=+1.1191；best Muon `muon_const`；best non-H0 baseline `signgd_const`；全 optimizer best `h0_ls`。
- `sample_k4`：Δ=+0.86324；best Muon `muon_const`；best non-H0 baseline `signgd_const`；全 optimizer best `h0_ls`。
- `top2_renormalized`：Δ=+0.680199；best Muon `muon_const`；best non-H0 baseline `gd_const`；全 optimizer best `h0_ls`。
- `onehot_mix_0.5`：Δ=+0.564585；best Muon `muon_const`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `top8_renormalized`：Δ=+0.527468；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `onehot_mix_0.25`：Δ=+0.523954；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `soft_copy_05`：Δ=+0.466888；best Muon `muon_const`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `random_context_column_permutation`：Δ=+0.440747；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `sample_k16`：Δ=+0.397955；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `onehot_mix_0.75`：Δ=+0.30727；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `ptb`：Δ=+0.134804；best Muon `muon_const`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `head_background_soft_target`：Δ=+0.0920439；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `context_gamma2`：Δ=+0.0795172；best Muon `muon_const`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `prototype_32`：Δ=+0.0710928；best Muon `muon_const`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `rewire_mix_0.5`：Δ=+0.0655429；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `association_0.5`：Δ=+0.0623513；best Muon `muon_const`；best non-H0 baseline `ngd_const`；全 optimizer best `h0_ls`。
- `tail_background_soft_target`：Δ=+0.0525219；best Muon `muon_const`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。
- `class_block_soft_target`：Δ=+0.0511009；best Muon `muon_ls`；best non-H0 baseline `signgd_ls`；全 optimizer best `h0_ls`。

Late 下的核心读法：Muon 的后期/final-loss 优势比 AUC 下更宽，尤其在 sparse target、frequency-matched soft target、top-k、onehot mixture、soft copy、PTB default、prototype/class-block/background soft target 中都能明显超过 non-H0 baselines。这个说明 Muon 的 benefit 很多时候不是只来自早期下降，而是能延续到 120-150 step 的后期窗口。

### 5.3 重要 caveat

- H0 inverse GD 在本实验中非常强。多数 setting 的全 optimizer best 是 `h0_ls` 或 `h0_const`，所以讨论 Muon benefit 时必须明确是否排除 H0。
- `hard_copy` 不支持“Muon 明显好”的结论：这个 setting 中 H0 constant 很强，而且没有 finite W*。
- `independent`、`rewired`、`flat_context` 这类去 association control 中，Muon 相对 scalar/no-H0 的优势通常很小；这支持 heavy-tail marginal alone 不足以解释大 Muon benefit。
- `permuted_entries` 中 Muon 的矩阵几何被打乱，Muon 相对 scalar/no-H0 没有优势或优势很弱；这支持 Muon 依赖有意义的矩阵结构。
- 所有 W* reference 都是独立 L-BFGS attempt。loss 图会画出 approximate `L(W*)` ref line，但这些 approximate reference 更适合作为 numerical landmark，而不是严格证明的全局最优。

## 6. 复现实验/刷新文档的常用命令

```powershell
python build_config_v2.py
python run_study_v2.py --retry-errors
python refresh_v2_outputs.py
python summarize_v2.py
python analyze_v2.py
python write_integrated_analysis_v2.py
python write_readme_v2.py
python write_readme_v2_cn.py
python validate_v2.py
```
