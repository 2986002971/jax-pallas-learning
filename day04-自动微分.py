# %% [markdown]
# # Day 4: 自动微分的二象性 (The Duality of Auto-Diff)
#
# **核心叙事**：自动微分（Auto-Diff）不是魔法，它是**线性代数**的工程化实现。它的本质是在切空间（Tangent Space）和伴随空间（Cotangent Space）之间构建桥梁。
# 我们将揭开雅可比矩阵（Jacobian Matrix）的面纱，从数学底层解释为什么前向模式（JVP）适合探索动力学系统，而反向模式（VJP）虽然是训练基石，却必须付出昂贵的显存代价。
#
# ## Part 1: 上帝视角 —— 雅可比矩阵与全景图
#
# 在讨论 JAX 怎么求导之前，我们需要先建立一个完整的神经网络训练图景。
#
# ### 1.1 系统全景：不仅仅是 $y=f(x)$
#
# 在一个深度学习系统中，我们将整个计算过程视为一连串变换的复合。假设一个 $L$ 层的网络：
#
# $$h_{l+1} = f_l(h_l, \theta_l)$$
# 其中：
#
# * $h_0 = x$: 输入数据。
# * $h_l$: 第 $l$ 层的中间激活值 (Activations)。
# * $\theta_l$: 第 $l$ 层的参数 (Weights & Biases)。
# * $y = h_L$: 最终输出。
# * $\mathcal{L} = \text{Loss}(y, y_{target})$: 标量损失函数。
#
# 在这个系统中，**每一个变量**（输入 $x$、参数 $\theta$、中间激活 $h$）都处于一个高维空间中。
#
# ### 1.2 雅可比矩阵 (The Jacobian)
#
# 对于任意一个变换 $f: \mathbb{R}^n \to \mathbb{R}^m$，其局部线性化由雅可比矩阵 $J \in \mathbb{R}^{m \times n}$ 描述：
#
# $$J_f(x) = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}$$
# **关键直觉**：
#
# 1. **它太大了**：在深度学习中，$n$ (参数量) 可能达到 $10^9$，$m$ (Loss) 通常是 1，但中间层的 $m$ (Feature Map) 也极大。显式构建 $J$ 是不可能的。
# 2. **它是局部的**：$J$ 的值依赖于当前的输入 $x$（这就是为什么非线性函数存在意义）。
# 3. **自动微分的本质**：我们不需要 $J$ 本身，我们只需要计算 **$J$ 与向量的乘积**。
#
# ## Part 2: JVP —— 推着切线走 (Forward-Mode)
#
# **数学定义**：**JVP (Jacobian-Vector Product)** 计算的是 $J \cdot v$。
#
# $$\text{Primal Out}, \text{Tangent Out} = f(x), \quad \frac{\partial f}{\partial x} \cdot v$$
#
# ### 2.1 几何与物理意义
#
# * **输入空间**：我们在点 $x$ 处，沿着方向 $v$ 移动了无穷小距离 $\epsilon$。
# * **输出空间**：$f(x)$ 会沿着哪个方向移动？移动多快？
# * **数学本质**：这就是**方向导数**。
#
# $$J \cdot v = \lim_{\epsilon \to 0} \frac{f(x + \epsilon v) - f(x)}{\epsilon} = \nabla_v f(x)$$
#
# ### 2.2 深度解析：JVP 与 ODE/动力学系统
#
# 在 ODE 中，我们研究状态随时间的演化 $\frac{dx(t)}{dt} = f(x(t), t)$。
# 当我们想知道初始状态的微小扰动 $\delta x(0)$ 如何随着时间传播到 $\delta x(t)$ 时，我们实际上是在求解变分方程（Variational Equation）：
#
# $$\frac{d}{dt}(\delta x(t)) = \frac{\partial f}{\partial x} \cdot \delta x(t) = \text{jvp}(f, (x(t),), (\delta x(t),))[1]$$
# 这就是为什么 JVP 在科学计算和敏感度分析中极其高效——它不需要反向传播，它是跟随时间箭头（前向）一起演化的。
#
# ### 2.3 JAX 实战：切空间的推演

# %%
import jax
import jax.numpy as jnp


# %%
# 一个模拟的复杂变换 R^3 -> R^2
def heavy_transform(x):
    # x: [3]
    # return: [2]
    return jnp.array([jnp.sin(x[0]) * x[1], jnp.exp(x[2] / (x[0] + 1e-5))])


# 1. Primal Point (当前参数/输入)
params = jnp.array([1.5, 2.0, 0.5])

# 2. Tangent Vector (切向量/扰动)
# 假设我们只对第一个参数施加单位扰动，看看对输出的影响
tangent_vector = jnp.array([1.0, 0.0, 0.0])

# 3. 计算 JVP
# jax.jvp 同时返回原函数值和切向量的投影
y, v_out = jax.jvp(heavy_transform, (params,), (tangent_vector,))

print(f"Primal Output (y) Shape: {y.shape}")  # (2,)
print(f"Primal Output (y): {y}")  # (2,)
print(f"Tangent Output (Jy): {v_out.shape}")  # (2,)
print(f"Tangent Output (Jy): {v_out}")  # (2,)

# v_out[0] 的含义：如果 x[0] 增加 epsilon，y[0] 会增加 v_out[0] * epsilon

# 告诉 make_jaxpr：第0个参数只作为静态配置，不要去 trace 它
print(
    "展示 JVP 的 Jaxpr:",
    jax.make_jaxpr(jax.jvp, static_argnums=(0,))(
        heavy_transform, (params,), (tangent_vector,)
    ),
)


# %% [markdown]
# 在 JVP 中，我们不仅仅是在计算函数的值，我们同时还在**并行**地维护和计算切线（Tangent / Differential）。
#
# 让我们像法医一样，逐行解剖这段 Jaxpr，看看链式法则是如何被翻译成汇编指令的。[[day02-追踪与编译]]
#
# #### 1. 变量映射表
#
# 首先，我们要分清谁是**原值 (Primal)**，谁是**切向量 (Tangent)**。
#
# * **输入端**:
# * `a`: 原始参数输入 x (Primal)。
# * `b`: 切向量输入 v (Tangent, 或者理解为 dx)。

# %% [markdown]
# * **切片 (Slicing)**:
# * `e` (来源于 `a`): 就是 x[0]。
# * `f` (来源于 `b`): 就是 v[0](即 dx[0]，输入在这个维度的扰动量)。
#

# %% [markdown]
# #### 2. 核心逻辑解剖：`sin(x[0])` 的微分
#
# 我们看这四行关键代码：

# %% language="bash"
# g:f32[] = sin e      # 1. 计算原函数值: sin(x[0])
# h:f32[] = cos e      # 2. 计算局部导数: cos(x[0])
# i:f32[] = mul f h    # 3. 链式法则应用: v[0] * cos(x[0])
#

# %% [markdown]
# 这里的数学原理非常直白：
# 假设 y=sin(u)，根据微分规则，dy=cos(u)⋅du。
#
# * **`g`**: 是 y (Primal Output)。
# * **`h`**: 是 cos(u) (Local Derivative)。
# * **`f`**: 是 du (Input Tangent / )。
# * **`i`**: 就是 dy (Output Tangent)。
#
# 所以，**`i` 正是 d(sin(x[0])) 的值**。JVP 就是这样把微分算子变成了一次简单的乘法。
#
# #### 3. 进阶：乘法法则 (Product Rule) 在哪？
#
# 原函数不仅是 `sin(x[0])`，而是 `sin(x[0]) * x[1]`。JVP 是如何处理乘法的微分  `(uv)′=u′v+uv′ `的？
#
# 往下看几行：

# %% language="bash"
# # ... (中间省略切片取出 x[1] 和 v[1]) ...
# l:f32[] = squeeze ... # x[1] (Primal)
# m:f32[] = squeeze ... # v[1] (Tangent)
#
# # 此时我们有四个变量：
# # g = sin(x[0])       (u)
# # i = v[0]*cos(x[0])  (u')
# # l = x[1]            (v)
# # m = v[1]            (v')
#
# n:f32[] = mul g l     # u * v   -> 原函数输出 (Primal Output)
# o:f32[] = mul i l     # u' * v  -> 乘法法则第一项
# p:f32[] = mul g m     # u * v'  -> 乘法法则第二项
# q:f32[] = add_any o p # u'v + uv' -> 最终的微分值 (Tangent Output)
#

# %% [markdown]
# ### 总结
#
# 你在 Jaxpr 里看到的每一个 `mul` 和 `add`，都是**具体的数值运算**。
#
# * **JVP 没有“反向传播”的图结构**，它只有**“伴随计算”**。
# * 只要算出了 `sin(x)`，JAX 就顺手算一下 `cos(x) * dx`。
# * 这一过程完全不需要保存 `sin(x)` 的值给后面用（那是 VJP 的事），算完这一步，前面的数据就可以扔掉了。
#
# 这就是为什么 JVP 极其省显存：**它是一次通过的，没有回头路。**
#
# ## Part 3: VJP —— 梯度的回溯与内存代价 (Reverse-Mode)
#
# **数学定义**：**VJP (Vector-Jacobian Product)** 计算的是 $w^T \cdot J$。
#
# $$\text{Adjoint Input} = w^T \cdot \frac{\partial f}{\partial x}$$
# 其中 $w$ 是输出空间的伴随向量（Cotangent Vector/Gradient）。
#
# ### 3.1 为什么训练需要 VJP？
#
# 对于 Loss Function $\mathcal{L}$，输出维度是 1。
# 我们想要计算梯度 $\nabla_x \mathcal{L}$。此时，$J$ 是一个 $1 \times n$ 的行向量。
# 如果我们令 $w = 1.0$ (即 $\frac{\partial \mathcal{L}}{\partial \mathcal{L}}$)，那么：
#
# $$1.0 \cdot J = [ \frac{\partial \mathcal{L}}{\partial x_1}, \dots, \frac{\partial \mathcal{L}}{\partial x_n} ]$$
# 一次 VJP 计算，就能得到所有输入的梯度。这比 JVP 高效得多（JVP 每次只能算一个方向的扰动）。
#
# ### 3.2 核心数学：为什么必须保存激活值？(The Memory Cost)
#
# 这是理解“训练显存远大于推理显存”的关键。
# 根据链式法则（Chain Rule），对于复合函数 $L = f_2(f_1(x))$：
#
# $$
# \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \underbrace{\frac{\partial f_2(y)}{\partial y}}_{J_2} \cdot \underbrace{\frac{\partial f_1(x)}{\partial x}}_{J_1} \quad (\text{其中 } y=f_1(x))
# $$
#
# 反向传播的过程，就是从左向右做矩阵乘法。然而，**雅可比矩阵 $J$ 的值通常依赖于输入**。
# 让我们看具体的算子：
#
# 1. **Sigmoid 激活**: $\sigma(x) = \frac{1}{1+e^{-x}}$
#    * 导数：$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$
#    * **代价**：为了在反向传播时算出导数，你必须知道 $\sigma(x)$ 的值（即前向传播的输出 $y$）。因此，**必须保存 $y$**。
# 2. **矩阵乘法 (Linear Layer)**: $y = W x$
#    * 对 $W$ 的梯度：$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T$
#    * **代价**：为了算出权重的梯度，你必须知道输入 $x$ 的值。因此，**必须保存 $x$**。
#
#    <details>
#    <summary>矩阵梯度推导与小例子</summary>
#
#    对于线性层 $y = Wx$，其中 $W \in \mathbb{R}^{m \times n}$，$x \in \mathbb{R}^n$，$y \in \mathbb{R}^m$，损失函数 $L$ 是标量：
#
#    损失 $L$ 对权重 $W_{ij}$ 的梯度由链式法则给出：
#
#    $$\frac{\partial L}{\partial W_{ij}} = \sum_{k=1}^{m} \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial W_{ij}}$$
#
#    由于 $y_k = \sum_{j=1}^{n} W_{kj} x_j$，所以：
#
#    $$\frac{\partial y_k}{\partial W_{ij}} = \begin{cases} x_j & \text{如果 } k=i \\ 0 & \text{如果 } k \neq i \end{cases}$$
#
#    因此：
#
#    $$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} x_j$$
#
#    将所有元素组合成矩阵形式：
#
#    $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T$$
#
#    其中 $\frac{\partial L}{\partial y} \in \mathbb{R}^{m}$ 是关于输出的梯度向量，$x^T \in \mathbb{R}^{1 \times n}$ 是输入向量的转置，得到的结果是与 $W$ 同形状的 $m \times n$ 梯度矩阵。
#
#    **小例子**（设 $L = y$）：
#    对于 $W \in \mathbb{R}^{1 \times 3}, x \in \mathbb{R}^{3 \times 1}$：
#
#    设 $W = [w_1, w_2, w_3], x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$，则 $y = w_1 x_1 + w_2 x_2 + w_3 x_3$。
#
#    当 $\frac{\partial L}{\partial y} = 1$ 时：
#    - $\frac{\partial L}{\partial W} = [x_1, x_2, x_3]$
#    - $\frac{\partial L}{\partial x} = \begin{bmatrix} w_1 \\ w_2 \\ w_3 \end{bmatrix}$
#
#    </details>
#
# 3. **ReLU**: $y = \max(0, x)$
#    * 导数：$1$ if $x > 0$ else $0$
#    * **代价**：你必须保存 $x$ 的正负符号（Mask）。
#
# 结论：
# 在前向传播（Forward Pass）中，我们是一边算一边扔（如果不做推断）。
# 但在准备反向传播（Backward Pass）时，JAX 必须把整个计算图路径上的关键Primal Values（输入或输出） 存入显存（HBM）。这就是 VJP 的代价。
#
# ### 3.3 JAX 实战：手动解剖 Backprop
#
# 我们已经知道反向传播需要保存中间状态。现在，我们不再纸上谈兵，而是用代码和编译器日志（Jaxpr）来**捉拿**这些被悄悄保存的变量。[[day02-追踪与编译]]
#
# #### **1. 代码实验：拆解 VJP**
#
# 首先，我们需要对比“自动挡”的 `grad` 和“手动挡”的 `vjp`。
#


# %%
def loss_fn(W, x):
    h = jnp.dot(W, x)  # 1. Matmul
    y = jax.nn.sigmoid(h)  # 2. Activation
    return 0.5 * jnp.sum(y**2)  # 3. Loss


# 初始化数据
W_key = jax.random.normal(jax.random.PRNGKey(0), (4, 5))
x_key = jax.random.normal(jax.random.PRNGKey(1), (5,))

# --- 方式 A: 自动挡 (grad) ---
# jax.grad 看起来像个黑盒，直接吐出梯度
grad_W = jax.grad(loss_fn)(W_key, x_key)

# --- 方式 B: 手动挡 (vjp) ---
# step 1: 前向传播 (Forward Pass)
# y_val 是 Loss 值
# vjp_fun 是一个"闭包"，它持有反向传播所需的全部"记忆"
y_val, vjp_fun = jax.vjp(loss_fn, W_key, x_key)

# step 2: 反向传播 (Backward Pass)
# 我们传入 dL/dL = 1.0，这把钥匙开启了回溯过程
grads = vjp_fun(1.0)
grad_W_manual = grads[0]

assert jnp.allclose(grad_W, grad_W_manual)
print("✅ 反向传播机制验证成功。")


# %% [markdown]
# #### **2. 显微镜下的 Jaxpr：寻找“显存杀手”**
#
# 现在，我们打印出计算图，看看 JAX 到底背着我们存了什么东西。

# %%
print("\n=== 1. JAX.GRAD (全景图) ===")
print(jax.make_jaxpr(jax.grad(loss_fn))(W_key, x_key))

print("\n=== 2. JAX.VJP (前向发货单) ===")
# static_argnums=(0,) 是为了告诉 JAX 第一个参数是函数，不要追踪
print(jax.make_jaxpr(jax.vjp, static_argnums=(0,))(loss_fn, W_key, x_key))

print("\n=== 3. VJP_FUN (反向闭包结构) ===")
print(jax.make_jaxpr(vjp_fun)(1.0))


# %% [markdown]
# #### **3. 深度解读：时间胶囊与闭包 (The Time Capsule)**
#
# 通过对比上面的输出，我们发现了 JAX 内存管理的真相：
#
# **A. `jax.grad` 的全景图 —— 融合 (Fusion)**
#
# 在 `grad` 的 Jaxpr 中，前向和反向被融合在了一起：

# %% language="bash"
# let
#     c = dot_general ...   # 前向计算
#     d = logistic c        # 前向激活 (Sigmoid Output)
#     # ... 中间省略 ...
#     f = mul d e           # 反向计算用到了 d
#     # ...
# in (o,)
#

# %% [markdown]
# * **现象**：变量 `d` (Sigmoid 输出) 在第 2 行生成，但直到倒数几行才被再次使用。
# * **代价**：这意味着 `d` 必须在显存中驻留整个计算周期，无法被提前释放。这就是长距离依赖带来的显存开销。
#
# **B. `jax.vjp` 的前向发货单 —— 存包 (Packing)**
#
# 观察 `jax.vjp` 的返回值（`in` 后面的部分），你会发现除了 Loss，还多了一堆东西：
#
# `in (k, a, b, f, i, 0.5)`
#
# * `k`: 唯一的 Primal Result (Loss 值)。
# * `a, b, f, i, 0.5`: 这些就是 **Residuals (残差)**。
# * `a`: 原始权重 （用于算输入梯度）。
# * `b`: 原始输入 （用于算权重梯度）。
# * `f`: Sigmoid 的导数辅助项。
# * `0.5`: 常数。
#

# %% [markdown]
# **核心洞察**：`jax.vjp` 返回时，并没有丢弃这些中间变量！它把这些变量打成了一个“时间胶囊”，交给了 Python 端的 `vjp_fun` 对象保管。
#
# **C. `vjp_fun` 的闭包验证 —— 取包 (Unpacking)**
#
# 最后，看 `vjp_fun` 的输入签名：
#
# `lambda a b c d e; f. let ...`
#
# * 分号 `;` 右边的 `f`：显式参数（传入的梯度 1.0）。
# * 分号 `;` 左边的 `a~e`：**隐式捕获参数 (Captured Arguments)**。
#
# **捉奸在床！** 左边的 `a~e` 正是上一阶段 `jax.vjp` 打包的那 5 个 Residuals。
# 这就解释了为什么你在 Python 里拿到的是一个 `function` 对象——这个对象内部持有着显存中那些 Tensor 的指针。只有当你调用这个函数结束时，这部分显存才会被释放。
#
# ---
#
# ### 总结：
#
# * **JVP** 是“阅后即焚”，算完就扔，显存极简。
# * **VJP** 是“打包带走”，必须把计算路径上的关键路标（Residuals）打包进闭包里，直到反向传播结束。
#
# 理解了这一点，你就理解了为什么 Transformer 的 `seq_len` 增加时，显存会线性爆炸——因为你要打包进闭包里的 Residuals 变多了。
#
# ### 🥚 彩蛋：JVP 的时间胶囊 —— jax.linearize
#
# **核心叙事**：我们刚才学了 vjp，它是把反向传播拆成了“前向 Pass”和“反向 Pullback”两步。那 **JVP** 能不能也拆开呢？
# 当然可以！这就是 jax.linearize。它是 JVP 的“分阶段”版本。
#
# #### 1. 场景：为什么要拆开 JVP？
#
# 想象你在做一个复杂的物理模拟 $y = f(x)$（非常耗时）。
# 现在你想知道：如果我在当前状态 $x$ 的基础上，往 $v_1, v_2, v_3, \dots$ 等 100 个不同方向稍微推一下，结果会怎么变？
#
# - **笨办法 (Standard JVP)**：
#   调用 100 次 `jax.jvp(f, (x,), (v_i,))`。
#   - *缺点*：你会把沉重的原函数 $f(x)$ 重复计算 100 次！
#
# - **聪明办法 (Linearize)**：
#   只算一次 $f(x)$，把这一点的“斜率信息”（雅可比性质）存下来。然后用这个缓存的“斜率发射器”去处理那 100 个方向。
#
# #### 2. 数学图示：矩阵的行与列
#
# 回顾雅可比矩阵 $J \in \mathbb{R}^{m \times n}$：
#
# - **VJP (jax.vjp)**：一次算出 $J$ 的**一行**（Row-wise）。适合 $m=1$ (Loss)。
# - **Linearize (jax.linearize)**：一次算出 $J$ 的**一列**（Column-wise）。适合 $n=1$ 或者我们需要探索输入空间的切向时。
#
# #### 3. 代码实战：牛顿法的加速器


# %%
def heavy_simulation(x):
    # 假设这是一个很重的计算
    # x: [3]
    return jnp.sin(x) * jnp.exp(x) + jnp.dot(x, x)


x_point = jnp.array([1.0, 2.0, 3.0])

# --- 方式 1: 普通 JVP (每次都要重新跑 heavy_simulation) ---
v1 = jnp.array([0.1, 0.0, 0.0])
v2 = jnp.array([0.0, 0.1, 0.0])
# jax.jvp(heavy_simulation, (x_point,), (v1,))  # 跑一次 f(x)
# jax.jvp(heavy_simulation, (x_point,), (v2,))  # 又跑一次 f(x) (浪费!)

# --- 方式 2: Linearize (只跑一次 f(x)) ---
# y_primal: f(x) 的值
# jvp_fun:  一个只包含线性映射逻辑的函数 ( f'(x) * v )
y_primal, jvp_fun = jax.linearize(heavy_simulation, x_point)

print(f"原函数值 (只算了一次): {y_primal}")

# 现在我们可以极其廉价地计算任意方向的切线投影
tan_1 = jvp_fun(v1)
tan_2 = jvp_fun(v2)

print(f"方向 v1 的扰动: {tan_1}")
print(f"方向 v2 的扰动: {tan_2}")

# %% [markdown]
# #### 4. 深度应用：一键生成整个雅可比矩阵

# %%
# 构造标准基向量 (1,0,0), (0,1,0), (0,0,1)
# 也就是 Identity Matrix
eye = jnp.eye(3)

# jvp_fun 本身一次只能处理一个向量 v
# 用 vmap 把它变成能处理一批向量 (即整个矩阵)
# in_axes=1 表示沿着 I 的列进行映射
jacobian_col_by_col = jax.vmap(jvp_fun, in_axes=1)(eye)

print("通过 linearize + vmap 还原的雅可比矩阵:")
print(jacobian_col_by_col)

# %% [markdown]
# ## Part 4: 梯度的阻断器 —— Stop Gradient
#
# 在 RL（如 Q-Learning）或 离散化技术（如 VQ-VAE）中，我们需要在计算图中人为地“切断”伴随向量的流动。
#
# ### 4.1 数学含义
#
# stop_gradient 算子 $\text{sg}(x)$ 定义为：
#
# * 前向传播：$\text{sg}(x) = x$ （恒等映射）
# * 反向传播：$\frac{\partial \text{sg}(x)}{\partial x} = 0$ （梯度阻断）
#
# 这等价于将变量 $x$ 视为**常数 (Constant)**，不参与链式法则的求导。
#
# ### 4.2 RL 场景：Bootstrapping
#
# 在更新 Q 网络时，目标值 $y = r + \gamma \max Q_{target}(s')$ 必须被视为常数。如果不切断梯度，优化器会试图通过改变 $Q_{target}$ 的参数来最小化误差，这是错误的（会导致训练发散）。

# %% [markdown]
# ## 总结：JAX 微分哲学的二象性
#
# Day 4 结束。这一天我们没有讲太多复杂的 API，而是回归了数学本源。
#
# 1. **JVP (前向)**：是 $J \cdot v$。它用于推断、微扰分析、ODE 求解。它不需要保存中间状态，内存极其高效。
# 2. **VJP (反向)**：是 $w^T \cdot J$。它用于训练。它利用链式法则高效求取所有参数的梯度，但代价是必须通过**保存激活值**来“记住”雅可比矩阵的状态。
# 3. **Stop Gradient**：是我们介入自动微分引擎的手术刀，用于控制梯度的流向。
#
# Day 5 预告：
# 明白了梯度的传递和显存的代价后，明天我们将徒手实现一个 Transformer Block。我们不再使用 Flax/Haiku，而是直接用 jax.jit, jax.vmap 和 jax.grad 管理参数字典 pytree。这将是你从“调包侠”进化为“架构师”的关键一步。
