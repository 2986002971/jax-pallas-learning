# %% [markdown]
# # Day 7: 突破高墙 —— Pallas 与内核编程艺术
#
# **核心叙事**：在之前的课程（Day 2）中，我们盛赞了 XLA 编译器的魔法。但在前沿的大模型开发中，我们撞上了一堵墙。今天，我们要学习如何翻越它。
#
# ## Part 1: 前情提要 —— 当自动挡遇到极限
#
# 大家还记得 **算子融合 (Operator Fusion)** 吗？[[day02-追踪与编译]]
#
# *   **没有 XLA**：`a = x + y`; `b = a * z`。GPU 会把 `a` 写回显存，再读出来算 `b`。
# *   **有 XLA**：编译器会把加法和乘法捏成一个大算子。数据读进芯片后，一口气算完 `(x+y)*z` 再写回。
#
# **既然 XLA 这么聪明，为什么我们还需要 Pallas？**
#
# 这就涉及到了 XLA 优化的**边界**。XLA 非常擅长处理 element-wise（逐元素）操作的融合，但面对涉及**复杂数据依赖**或**巨大中间结果**的算法（典型的如 Attention）时，通用的启发式算法往往会失效。
#
# ### 1.1 房间里的大象：中间显存物化 (Intermediate Materialization)
#
# 以标准 Attention 为例：$Attention(Q, K, V) = \text{softmax}(Q K^T) V$。
#
# 1.  计算 $S = Q K^T$。如果序列长度是 4096，这是一个 $4096 \times 4096$ 的巨大矩阵。
# 2.  XLA 虽然能融合简单的加减乘除，但面对这种矩阵乘法接 Softmax，它通常不得不把这个巨大的 $S$ **完整地写回 HBM**。
# 3.  然后再从 HBM 读取 $S$ 来算 Softmax。
#
# **这就是瓶颈所在**：虽然算子内部融合了，但算子之间巨大的中间结果撑爆了显存带宽（甚至撑爆显存容量）。[[day06-计算的物理形态]]
#
# ---
#
# ### 1.2核心隐喻 —— 修正后的厨房模型
#
# 为了更准确地理解 Pallas 要解决的问题，我们用厨房与做菜来比喻。
#
# *   **HBM (显存)** = 远处的巨大**冷库**。容量大，但存取慢。
# *   **SRAM (片上缓存)** = 灶台边极小的**案板**。就在手边，速度极快，但空间很小。[[day06-计算的物理形态]]
#
# #### **Scenario A: 简单的做菜 (XLA 的强项)**
#
# **任务**：把土豆削皮，然后切块。
#
# *   **XLA 做法 (Smart Fusion)**：它不会削完皮跑回冷库放好，再跑回来拿去切。它会把土豆拿来放在案板上，**一口气**削皮+切块，然后放回冷库。
# *   **评价**：这叫“算子融合”，XLA 做得很完美，你不需要插手。
#
# #### **Scenario B: 复杂的宴席 (XLA 的弱项)**
#
# **任务 (Attention)**：你需要准备 1000 人份的混合沙拉（这就好比那个巨大的 $Q K^T$ 矩阵）。
#
# *   **XLA 做法 (OOM Risk)**：它试图先把 1000 人份的菜全部切好，堆满整个厨房（显存爆炸），然后再开始拌沙拉酱。如果案板（SRAM）放不下，它就只能把切好的菜先运回冷库（HBM），要拌的时候再运回来。**大量的带宽浪费在运送半成品上。**
#
# *   **Pallas 做法 (Tiling)**：你强制规定流程——“一次只做 1 人份”。
#     1.  去冷库拿 1 人份原料。
#     2.  在案板上切好、拌好、装盘。
#     3.  直接端出去。
#     4.  重复 1000 次。
#
# **结果**：你永远不需要一个能装下 1000 人份沙拉的巨大容器。案板（SRAM）虽小，但利用率极高。这就是 **Pallas** 允许你做的事情——手动控制这个“切分”和“流水线”的过程。
#
# ---
#
# ### 1.3: 理论升华 —— 微观世界的 Sharding
#
# 如果这种“切分”听起来很眼熟，那是因为你已经在 **Day 3** 学过它了。[[day03-自动向量化]]
#
# **Pallas 其实就是微观世界的 Sharding。**
#
# 这两种并行的哲学是惊人一致的，呈现出一种分形之美：
#
# | 特性 | **Day 3: Distributed Sharding** | **Day 7: Pallas Kernel** |
# | :--- | :--- | :--- |
# | **战场** | **宏观** (跨 GPU/TPU 设备) | **微观** (单芯片内部) |
# | **瓶颈** | 网线/NVLink 通信带宽 | HBM 显存带宽 |
# | **高速区** | 单个 GPU 的显存 | 芯片核心的 SRAM |
# | **切分对象** | Global Batch / Model 维度 | Tensor 的 Block / Tile 维度 |
# | **工具** | `PartitionSpec` / `Mesh` | `BlockSpec` / `Ref` |
# | **核心逻辑** | **少跨卡通信，多本地计算** | **少读写 HBM，多在 SRAM 计算** |
#
# *   **Day 3** 教你如何把数据切碎了塞进不同的**显卡**。
# *   **Day 7** 教你如何把显存里的数据切碎了塞进**SRAM**。
#
# ---
#
# ### 1.4: 编程范式 —— 从“指针”到“视图”
#
# 既然要在微观世界做“切分”，具体的编程手感如何呢？
#
# 如果你了解 **Triton**（OpenAI 开发的 GPU 语言），你会发现 Pallas 的目标与它一致，但思维方式略有不同：
#
# *   **CUDA/Triton (指针流)**：
#     你需要自己计算显存的**物理指针偏移量**。你需要算：“我是第几个线程，我要偏移多少个字节，去读取哪一块数据。”这是一切都要亲力亲为的“中央厨房”。
#
# *   **Pallas (视图流)**：
#     这更像是 JAX 的风格。你不需要算指针，你只需要定义**切分规则 (BlockSpec)**。
#     *   编译器会自动把大张量切好，送进 SRAM。
#     *   你的 Kernel 拿到的直接是一个**引用 (Ref)**，就像一个在这个小切片上的 Numpy 数组。
#
# **一句话总结**：在 Pallas 里，我们像在 Day 3 定义 `PartitionSpec` 一样定义 `BlockSpec`，剩下的搬运工作，交给流水线自动完成。

# %%
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax import custom_vjp

# 设置随机种子
key = jax.random.PRNGKey(42)


# %% [markdown]
# ---
# ## Part 2: 定义前向 Kernel (The Chef)
#
# 我们先定义一个最简单的矩阵加法内核。
#
# **注意**：这个函数**不知道**它处理的数据有多大。它只知道它手里拿到了一小块 `Ref`。这使得 Kernel 代码与数据规模解耦。
#
#

# %%
def add_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]


# %% [markdown]
# ---
# ## Part 3: 定义调度策略 (The Logistics)
#
# 接下来，我们需要定义如何把一个大矩阵切分给上面的小内核。
#
# 假设我们有一个 `(1024, 1024)` 的大矩阵，我们想用 `(128, 128)` 的块去处理它。

# %%
# 定义分块大小 (Block Size)
BLOCK_SHAPE = (128, 128)

# 创建 BlockSpec
# 输入输出的切分逻辑是一样的，所以我们共用这个 Spec
common_spec = pl.BlockSpec(index_map=lambda i, j: (i, j), block_shape=BLOCK_SHAPE)


# %% [markdown]
# ---
# ## Part 4: 封装前向调用
#
# 我们现在把 Kernel 和 Spec 组装起来，创建一个普通的 Python 函数 `pallas_add`。
#
#

# %%
def pallas_add(x, y):
    # 1. 动态计算 Grid 大小
    grid = (x.shape[0] // BLOCK_SHAPE[0], x.shape[1] // BLOCK_SHAPE[1])
    print(f"Computed Grid: {grid}")

    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(common_spec, common_spec),
        out_specs=common_spec,
        # interpret=True,  # 先用解释器验证，没问题后再关掉
    )(x, y)


# 测试一下前向传播
x_dummy = jax.random.normal(key, (1024, 1024))
y_dummy = jax.random.normal(key, (1024, 1024))

print(f"Input type: {x_dummy.dtype}, shape: {x_dummy.shape}")
# 这一步应该能运行，但还不能求导
result = pallas_add(x_dummy, y_dummy)
print(f"Error vs JAX: {jnp.max(jnp.abs(result - (x_dummy + y_dummy)))}")

# %% [markdown]
# ---
# ## Part 5: 进阶 —— 自定义反向传播 (Custom VJP)
#
# 如果你现在尝试 `jax.grad(pallas_add_forward)(x, y)`，会报错。
# %%
try:
    jax.grad(pallas_add)(x_dummy, y_dummy)
except Exception as e:
    print("Expected Error during grad:", e)
# %% [markdown]
# 因为 Pallas Kernel 内部包含了副作用（原地写入 `z_ref`），JAX 的自动微分引擎无法追踪。[[day02-追踪与编译]]
#
# 我们必须手写反向传播逻辑。
#
# ### 5.1 数学原理
# 对于 $z = x + y$：
# * 已知：输出的梯度 $\frac{\partial L}{\partial z}$ (我们称为 `dz` 或 `grad_output`)。
# * 求解：$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot 1 = dz$
# * 求解：$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot 1 = dz$
#
# 所以反向传播非常简单：把收到的梯度 `dz` 原样复制给 `dx` 和 `dy` 即可。
#
# ### 5.2 定义反向 Kernel
# 虽然只是简单的复制，但我们依然要用 Pallas Kernel 来实现它，以展示流程。
#
#

# %%
def add_backward_kernel(dz_ref, dx_ref, dy_ref):
    """
    反向 Kernel
    输入: dz_ref (从上一层传回来的梯度)
    输出: dx_ref, dy_ref (计算出的关于输入的梯度)
    """
    # 1. 读取输出梯度
    dz = dz_ref[...]

    # 2. 计算输入梯度 (对于加法，梯度就是直接传播)
    # 注意：在复杂的算子中，这里可能需要 atomic_add
    # 但因为我们是一对一的 Block 映射，没有重叠写入，所以直接赋值即可
    dx_ref[...] = dz
    dy_ref[...] = dz


# %% [markdown]
# ### 5.3 绑定 VJP (The Wiring)
#
# 我们使用 `jax.custom_vjp` 将前向和反向逻辑缝合在一起。[[day04-自动微分]]
#

# %%
# 1. 定义对外的主函数接口
@custom_vjp
def matrix_add(x, y):
    return pallas_add(x, y)


# 2. 定义前向逻辑 (Forward)
# 返回值必须是: (输出, 残差)
# 残差 (Residuals) 是为了反向传播时使用的数据。[[day04-自动微分]]
# 对于加法，我们不需要保留 x 或 y 就能算梯度，所以残差为空。
def matrix_add_fwd(x, y):
    return pallas_add(x, y), ()  # Empty residuals


# 3. 定义反向逻辑 (Backward)
# 输入: (残差, 输出梯度)
# 输出: (x_grad, y_grad)
def matrix_add_bwd(res, dz):
    # 这里我们定义反向的调用逻辑
    grid = (dz.shape[0] // BLOCK_SHAPE[0], dz.shape[1] // BLOCK_SHAPE[1])

    # 调用反向 Kernel
    # 注意：输入是 dz，输出是 dx, dy
    dx, dy = pl.pallas_call(
        add_backward_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(dz.shape, dz.dtype),  # dx shape
            jax.ShapeDtypeStruct(dz.shape, dz.dtype),  # dy shape
        ],
        grid=grid,
        in_specs=[common_spec],  # dz
        out_specs=[common_spec, common_spec],  # dx, dy
    )(dz)

    return dx, dy


# 4. 正式绑定
matrix_add.defvjp(matrix_add_fwd, matrix_add_bwd)


# %% [markdown]
# ---
# ## Part 6: 验证时刻
#
# 现在，我们拥有了一个**支持自动微分**的高性能自定义 Pallas 算子！
# 让我们来验证它的梯度是否正确。
#
#

# %%
def loss_fn(x, y):
    # 计算 z = x + y
    z = matrix_add(x, y)
    # Loss = sum(z)
    return jnp.sum(z)


# 1. 准备数据
x = jax.random.normal(key, (1024, 1024))
key, _ = jax.random.split(key)
y = jax.random.normal(key, (1024, 1024))

# 2. 计算梯度 (Pallas implementation)
grad_x_pallas, grad_y_pallas = jax.grad(loss_fn, argnums=(0, 1))(x, y)

# 3. 计算标准答案 (JAX Native)
grad_x_jax, grad_y_jax = jax.grad(lambda a, b: jnp.sum(a + b), argnums=(0, 1))(x, y)

# 4. 对比差异
diff = jnp.max(jnp.abs(grad_x_pallas - grad_x_jax))
print(f"Gradient Difference: {diff:.6f}")

assert diff < 1e-4, "梯度计算有误！"
print("Success! Pallas VJP is working correctly.")
