# %% [markdown]
# **Day 3: 维度折叠与分布式重构 (Vmap & Sharding)**
#
# **核心叙事**：从“微观向量化”到“宏观分布式”。我们不再编写并行的代码，我们只编写并行的**数据类型**。
#
# * **Day 3.1: Vmap (自动向量化)** —— 在单卡内，消灭 For 循环。
# * **Day 3.2: The New Sharding (Mesh & PartitionSpec)** —— 在多卡间，定义数据的物理形态。
# * **Day 3.3: 显微镜下的 HLO** —— 验证通信算子的自动生成。
#
# ---
#
# ### Part 1: Vmap —— 维度的魔法师 (Micro-Parallelism)
#
# **概念**：`vmap` (Vectorizing Map) 是 JAX 的“单机大杀器”。它将处理单个样本的函数，自动转换为处理整个 Batch 的函数。
#
# #### **1. in_axes 的维度舞蹈**
#
# `vmap` 的精髓在于 `in_axes`。你必须极其清楚哪个参数需要切分（Batch），哪个参数需要广播（Weights）。

# %%
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_num_cpu_devices", 8)


# %%
# 1. 定义核心逻辑 (只写处理单个样本的数学公式)
# 这是一个简单的线性层: y = w @ x + b
def predict_single(w, b, x):
    return jnp.dot(w, x) + b


# 2. 准备数据 (模拟参数和 Batch 数据)
# W, b 是共享参数，x 是 Batch 输入
W = jax.random.normal(jax.random.PRNGKey(0), (10, 5))  # (Out, In)
b = jax.random.normal(jax.random.PRNGKey(1), (10,))  # (Out,)
batch_x = jax.random.normal(jax.random.PRNGKey(2), (32, 5))  # (Batch, In)

# 3. 【核心】映射规则
# in_axes=(None, None, 0)
# -> W: None (不切，所有人共用一份)
# -> b: None (不切，所有人共用一份)
# -> x: 0    (沿着第0维切开，每个虚拟线程拿一个样本)
batched_predict = jax.vmap(predict_single, in_axes=(None, None, 0))

# 4. 运行
res = batched_predict(W, b, batch_x)
print("Single Output Shape: (10,)")
print(f"Vmap   Output Shape: {res.shape}")  # (32, 10)


# %% [markdown]
# #### **2. Jaxpr 里的秘密**
#
# 怎么证明 JAX 不是在跑循环？看 Jaxpr。[[day02-追踪与编译]]

# %%
print("\n>>> Vmap 后的 Jaxpr (注意 dot_general):")
print(jax.make_jaxpr(batched_predict)(W, b, batch_x))


# %% [markdown]
# **解读**：你会看到 Python 的循环完全消失了，取而代之的是 `dot_general` 操作，它直接在张量层面处理了额外的维度。这是 SIMD（单指令多数据）的胜利。
#
# ---
#
# ### 间章：从数学公式到工程代码——数据的“躺平”哲学
#
# 在进入具体的神经网络层实现（Part 2）之前，我们需要先解决一个让无数初学者（甚至老手）头疼的问题：**为什么代码里的矩阵乘法顺序，好像和数学书上写的不太一样？**
#
# 这涉及到计算机科学与纯数学之间的一个经典“对称破缺”：**行优先与列优先**。
#
# #### 1. 教科书 vs. 计算机：向左走，向右走
#
# *   **纯数学传统（列向量主导）**
#     在经典的线性代数教科书中，向量 $\mathbf{x}$ 默认是“站着”的（**列向量**，$d \times 1$）。因此，当我们对数据做变换（用矩阵 $W$ 乘它）时，矩阵必须放在左边：
#     $$ \mathbf{y} = W \mathbf{x} + \mathbf{b} $$
#     这种写法符合人类从左到右阅读“算子作用于对象”的直观逻辑。
#
# *   **机器学习/NumPy 传统（行优先主导）**
#     然而，在 Python (NumPy/JAX/PyTorch) 的世界里，情况反过来了。这是因为：
#     1.  **内存布局**：NumPy 默认采用 C-style 的**行优先（Row-major）**存储。这意味着读取一行数据的速度通常比读取一列要快（缓存命中率高）。
#     2.  **为了“批处理”（Batch）**：在 ML 中，我们很少只处理一个样本，而是一次处理 $N$ 个样本。如果我们把每个样本看作一行，堆叠起来就形成了一个 $N \times D$ 的矩阵 $\mathbf{X}$。
#         $$ \mathbf{X} = \begin{bmatrix} — \text{sample 1} — \\ — \text{sample 2} — \\ \vdots \end{bmatrix} $$
#         这里的 $\mathbf{X}$ 的每一行，就是一个“躺平”的行向量。
#
# 为了适应这种 $N \times D$ 的数据格式，我们的矩阵乘法必须变成：
# $$ \mathbf{Y} = \mathbf{X} @ \mathbf{W} + \mathbf{b} $$
# 注意到了吗？**$W$ 跑到了右边，而且它的形状通常和数学定义中的转置了**。
#
# #### 2. @ 运算符的“潜规则”与对称破缺
#
# 在 NumPy（以及 JAX/PyTorch）中，`@` 运算符（即 `matmul`）不仅仅是简单的点积，它有一套严格的**对齐规则**，这其实就承认了“最后一维是行”的地位：
#
# > **规则：A @ B，是用 A 的最末维 去乘 B 的倒数第二维。**
#
# *   如果 $A$ 是 $(N, \color{red}{D})$
# *   那么 $B$ 必须是 $(\color{red}{D}, M)$
# *   结果是 $(N, M)$
#
# 这就解释了为什么当 $X$ 在左边时，它被视为由无数个行向量组成的集合。在这种运算逻辑下，**“第一维是索引（第几个样本），最后一维是内容（样本的特征）”** 成了默认公理。
#
# #### 3. 广播机制（Broadcasting）：`jnp.dot(w, x) + b` 为什么可行？
#
# 你会在一些简单示例（例如上文part1）中看到类似 `jnp.dot(w, x) + b` 的写法，这看似更像数学公式 $Wx+b$，它是怎么工作的？这得益于 Python 强大的**灵活度（自动推断）与广播（Broadcasting）**。
#
# 假设我们处理**单个样本**：
# *   $w$ (权重): 形状 `(out, in)`
# *   $x$ (输入): 形状 `(in,)` —— 注意，它是 1D 数组。
# *   $b$ (偏置): 形状 `(out,)`
#
# 当我们执行 `jnp.dot(w, x)`：
# 1.  NumPy/JAX 看到 $x$ 在右边，且是 1D，**自动**将其视为列向量 `(in, 1)`（但在计算后自动降维回 1D）。
# 2.  计算结果形状为 `(out,)`。
#
# 接下来执行 `+ b`：
# *   结果 `(out,)` 加 偏置 `(out,)`。
# *   形状完美匹配（Element-wise add）。
#
# **但是！** 一旦我们进入真正的机器学习实战（Part 2），我们几乎总是处理 **Batch**（批次数据）。
# *   输入 $X$ 变成了 `(Batch_Size, in)`。
# *   此时 `jnp.dot(w, X)` 就会报错或产生错误的维度含义。
# *   我们必须切换到 **$X @ W$** （或 `X @ W.T`）的形式，让 $X$ 在左边掌控全局。
#
# #### 4. 总结：机器学习中的矩阵形状约定
#
# 为了顺利进入下一章节，请牢记这套**“工程领域的默认形状”**：
#
# | 实体 | 数学符号 (Math) | 代码形状 (Code) | 备注 |
# | :--- | :--- | :--- | :--- |
# | **输入数据 (Batch)** | $X$ | `(N, D_in)` | $N$ 个样本，每个样本 $D_{in}$ 维特征，**行优先** |
# | **权重矩阵 (Weights)** | $W$ | `(D_in, D_out)` | 为了配合 $X @ W$，通常不仅要在右边，形状也要转置 |
# | **偏置向量 (Bias)** | $b$ | `(D_out,)` | 通过广播机制加到每一行上 |
# | **核心运算** | $y = Wx + b$ | `Y = X @ W + b` | **这是最重要的公式转换！** |
#
# 理解了这一点——**数据是躺着的，运算是右乘的**——你就理解了所有深度学习框架（TensorFlow, PyTorch, JAX）设计 API 的底层逻辑。
#
# 接下来，我们在 Part 2 中，就将利用这个 `X @ W + b` 的范式，来通过代码手写一个真正的全连接层。
#
# #### 🍻 彩蛋：不想背维度规则？召唤“爱因斯坦求和约定”
# <details>
# 如果你觉得上面的“最后一维”、“倒数第二维”、“广播规则”听起来让人头大，NumPy 和 JAX 提供了一个**上帝视角**的工具，让你直接跳过这些隐式规则，用最接近数学公式的方式写代码。
#
# 这就是 **`jnp.einsum` (Einstein Summation Convention)**。
#
# 还记得我们说 Python 里的矩阵乘法是 $Y = X @ W$ 吗？
# *   $X$ 的形状是 `(batch, in_dim)`
# *   $W$ 的形状是 `(in_dim, out_dim)`
# *   $Y$ 的形状是 `(batch, out_dim)`
#
# 用 `einsum`，你甚至不需要思考谁在左谁在右，只需要把**索引的名字**告诉它：
#
# ```code
# # 'bi' 代表 X 的维度 (batch, in_dim)
# # 'io' 代表 W 的维度 (in_dim, out_dim)
# # '->' 代表输出
# # 'bo' 代表输出 Y 的维度 (batch, out_dim)
#
# Y = jnp.einsum('bi, io -> bo', X, W)
# ```
#
# **这一行代码发生了什么？**
# 它在告诉 JAX：“找到所有相同的索引（这里是 `i`），把它对应的维数乘起来并求和（Sum reduction），剩下的索引（`b` 和 `o`）保留下来作为输出的维度。”
#
# **为什么说它是“神器”？**
# 1.  **显式优于隐式**：你再也不用担心自动广播会悄悄搞错维度。你想让哪两维相乘，就给它们写一样的字母。
# 2.  **无视形状顺序**：
#     *   如果此时 $W$ 是转置过的 `(out_dim, in_dim)` 怎么办？
#     *   普通写法：`X @ W.T` (还得想一下要不要转置)
#     *   Einsum 写法：`jnp.einsum('bi, oi -> bo', X, W)` (直接把索引对应上，JAX 自己会去管内存怎么读)
# 3.  **JAX 的最爱**：在 JAX 的底层编译器（XLA）眼中，`einsum` 通常能被优化成极高效率的计算图，有时候比你手写的 `transpose` + `matmul` 还要快。
#
# 所以，如果你在未来更复杂的 Attention 机制实现中迷失在维度的海洋里，记得回来试试这把“手术刀”。
# </details>
# ---
#
# ### Part 2: Sharding —— 拥抱 Shardy 新范式 (Macro-Parallelism)
#
# **背景重构**：
#
# * **过去 (pmap)**：手动管理设备 ID，手动写 `pmean`。代码和单机版代码不同，难以维护。
# * **过渡 (GSPMD)**：编译器自动插入通信，但有时像“黑盒”，不知道它到底怎么切的。
# * **现在 (Shardy & Explicit Sharding)**：基于 `Mesh` 和 `PartitionSpec` 的显式控制。代码即单机代码，但**数据类型携带了物理位置信息**。
#
# #### **1. 搭建舞台：模拟 8 卡环境**
#
# 为了演示，我们需要“欺骗”XLA，把CPU切成八份。

# %%
# 在上方import jax之前已经声明了八个设备
# 创建物理网格：2行4列
# axis_names 是我们给物理轴起的逻辑名字
# 'data': 用于数据并行
# 'model': 用于模型/张量并行
devices = mesh_utils.create_device_mesh((2, 4))
mesh = Mesh(devices, axis_names=("data", "model"))

print(f"设备网格:\n{mesh}")


# %% [markdown]
# #### **2. PartitionSpec 的策略艺术**
#
# 结合 **算术强度 (Arithmetic Intensity)** 理论。
#
# * **Data Parallel (DP)**: 算力/通信比高。Batch 维切分，参数复制。
# * **Tensor Parallel (TP)**: 算力/通信比低（需高频通信）。参数切分。

# %%
# 假设我们有一个巨大的矩阵乘法: Y = X @ W
# X: (Batch=64, Seq=128)
# W: (Seq=128, Hidden=256)

# 定义数据的逻辑切分规则 (PartitionSpec)
# P('data', 'model') 意味着：
#   第 0 维 (Batch) 被切分到 'data' 轴 (2份)
#   第 1 维 (Seq)   被切分到 'model' 轴 (4份)
spec_X = P("data", "model")

# 权重 W 做张量并行 (TP)
# 第 0 维 (Seq) 必须和 X 的第 1 维对齐，所以也切分到 'model' 轴
# 第 1 维 (Hidden) 不切分 (None)
spec_W = P("model", None)

# 输出 Y (Batch, Hidden)
# Batch 维依然在 'data' 轴，Hidden 维没切
spec_Y = P("data", None)


# %% [markdown]
# #### **3. Shardy 的“计算跟随数据”**
#
# 这是 JAX 2026 的核心哲学：**Compile-led Parallelism**。你不需要告诉函数怎么并行，你只需要告诉它输入和输出长什么样。

# %%
# 生成数据
X = jax.random.normal(jax.random.PRNGKey(0), (64, 128))
W = jax.random.normal(jax.random.PRNGKey(1), (128, 256))


# 定义计算逻辑 (完全是单机代码！)
def matmul_fn(x, w):
    return jnp.dot(x, w)


# 【核心】注入 Sharding 约束
# 告诉 JIT：输入遵循 spec_X/W，输出必须满足 spec_Y
sharded_matmul = jax.jit(
    matmul_fn,
    in_shardings=(NamedSharding(mesh, spec_X), NamedSharding(mesh, spec_W)),
    out_shardings=NamedSharding(mesh, spec_Y),
)

# 此时，Shardy 分区器介入，计算如何插入通信算子
result = sharded_matmul(X, W)

# 验证结果的分片属性
print(f"Result Sharding: {result.sharding}")
# 你会看到它严格遵守了 P('data', None)


# %% [markdown]
# ---
#
# ### Part 3: 显微镜下的 HLO (验证通信)
#
# *不要轻信魔法。我们要像法医一样解剖代码，看看编译器到底有没有帮我们做 All-Reduce。*[[day02-追踪与编译]]

# %%
# Lowering 到 HLO (High Level Optimizer) 代码
lowered = sharded_matmul.lower(X, W)
compiled = lowered.compile()
hlo_text = compiled.as_text()

print("\n>>> HLO 代码展示:")
print("\n", hlo_text)  # 打印前

print("\n--- HLO 通信指令侦测 ---")
# 在 HLO 中寻找 collective ops
ops_to_find = ["all-reduce", "all-gather", "reduce-scatter"]

found_ops = [op for op in ops_to_find if op in hlo_text]

if found_ops:
    print(f"✅ 成功！发现通信指令: {found_ops}")
    print("解释：由于我们在 'model' 轴切分了收缩维 (Seq)，")
    print("XLA 必须在计算后执行 All-Reduce 以聚合局部结果。")
else:
    print("❌ 未发现通信指令。")


# %% [markdown]
# #### **进阶：为什么不用 `pmap` 了？**
#
# `pmap` 是“手动档”，它强制你写出针对**单个设备**的代码。而现在的 `Mesh` + `jit` 是“自动驾驶”。
#
# * 在 `pmap` 里，做复杂的张量并行（TP）简直是噩梦，你需要自己算索引。
# * 在 `GSPMD/Shardy` 里，你只要定义 `P('model', None)`，编译器自己去算该怎么切。
#
# ---
#
# ### Part 4: 补充材料与延伸阅读 (Advanced)
#
# **给学有余力的同学：**
#
# 1. **shard_map (手动挡的回归)**：
# 虽然 `jit` 自动推导很强，但有时候我们需要极致的控制（比如在 Ring Attention 中手动控制通信环）。这时不要怀念 `pmap`，请使用 **`jax.experimental.shard_map`**。它是新一代的显式并行 API，允许你在 Mesh 上手写通信原语。
# 1. **Shardy vs GSPMD**：
# 你今天看到的 `NamedSharding` 在旧版本底层是 GSPMD，但在 JAX 2026 版本中已切换为 **Shardy**。它的区别在于：Shardy 在中间表示层（StableHLO）保留了轴的名字（如 'model'），而不是把它们变成冷冰冰的数字索引。这对 Debug 极其重要。
# 1. **算术强度 (Arithmetic Intensity)**：
# 为什么我们要搞这么复杂的切分？因为 **计算比通信快太多了**。在 TPU v6e 上，这个比例高达 5000:1。我们的目标就是：**少通信，多计算**。
#
# ---
#
# ### 总结
#
# Day 3 结束，你已经掌握了：
#
# 1. **Vmap**：让单卡代码自动 Batch 化。
# 2. **Mesh & PartitionSpec**：用描述数据的方式，实现数据并行和模型并行。
# 3. **HLO Analysis**：看穿编译器的把戏，确认 `all-reduce` 的存在。
#
# **Day 4 预告**：现在我们会并行计算了，但怎么**并行求导**？JAX 的自动微分引擎 `jax.grad` 到底是如何处理这些分布式数组的梯度的？它比 PyTorch 的 Autograd 强在哪里？明天见！
