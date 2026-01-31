# %% [markdown]
# # Day 9: 拆解巨石 —— Online Softmax 与分块流式处理
#
# **核心叙事**：在上一节的讨论中，我们发现如果 Vocab Size 过大（例如 128k），SRAM（通常几十 KB 到几 MB）根本塞不下完整的一行。
# 今天我们要实现 **Online Safe Softmax**。这不仅是解决显存限制的方案，更是通往 FlashAttention 的必经之路。
#
# ## Part 1: 数学原理 —— 如何“流式”寻找最大值与和
#
# 假设我们将一行数据切成两块：$x = [x_1, x_2]$。
#
# ### 1.1 传统做法 (Offline)
# 必须拿到 $x_1, x_2$ 的全部数据，算出全局 $m = \max(x_1, x_2)$，然后求和。
#
# ### 1.2 Online 做法 (Rescaling Trick)
# Milakov & Gimelshein 提出的在线算法允许我们逐步更新统计量。
#
# 令 $m_{old}$ 为旧的最大值，$d_{old}$ 为旧的指数和（Denom）。
# 当新数据块 $x_{new}$ 进来时，局部最大值是 $m_{block}$。
#
# 1.  **更新最大值**：
#     $$m_{new} = \max(m_{old}, m_{block})$$
# 2.  **更新指数和 (关键)**：
#     我们不能直接加，因为旧的和是基于 $e^{x-m_{old}}$ 算的，而我们需要基于 $e^{x-m_{new}}$。
#     需要做一个修正（Rescale）：
#     $$d_{new} = d_{old} \times e^{m_{old} - m_{new}} + \sum e^{x_{new} - m_{new}}$$
#
# 这个 $e^{m_{old} - m_{new}}$ 就是**修正系数**。如果 $m_{new}$ 没变，系数是 1；如果发现了更大的数，系数 $<1$（相当于把旧的求和结果缩小）。
#
# ---
#
# ## Part 2: Pallas 实现策略 (Block-wise Loop)
#
# 为了模拟“显存放不下”的场景，我们将输入数据 reshape 增加一个维度，强制 Kernel 每次只处理一小块。
#
# *   **逻辑输入**：`(Batch, Vocab)`
# *   **物理输入**：`(Batch, Num_Blocks, Block_Size)`
# *   **SRAM 限制**：每次只能读入一个 `Block_Size`。
#
# **算法流程 (2-Pass)**：
# 1.  **Loop 1 (Statistics)**: 遍历所有 Blocks，利用 Rescaling Trick 计算全局 `Max` 和 `Sum`。
# 2.  **Loop 2 (Normalize)**: 再次遍历所有 Blocks，读取数据，除以全局 Sum，写入结果。
#
# ### 2.1 准备环境与数据

# %%
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

# 模拟配置
BATCH = 32
VOCAB = 8192  # 假设这是总长度
BLOCK_SIZE = 1024  # SRAM 只能放下这么多
NUM_BLOCKS = VOCAB // BLOCK_SIZE

# 构造分块后的输入
# Shape: (Batch, Num_Blocks, Block_Size)
x_sim = jax.random.normal(jax.random.PRNGKey(42), (BATCH, NUM_BLOCKS, BLOCK_SIZE))


# %% [markdown]
# ### 2.2 编写 Online Softmax Kernel
#
# 这个 Kernel 稍微有点复杂，因为它内部包含两个循环。
# **注意**：`x_ref` 这里虽然是指向整个 Batch 的一行（包含所有 Blocks），但在 Kernel 内部我们通过循环切片来模拟“流式读取”。[[day07-pallas初探]]
#

# %%
def online_softmax_kernel(x_ref, o_ref):
    """
    x_ref: (N_BLOCKS, BLOCK_SIZE) [in HBM]
    o_ref: (N_BLOCKS, BLOCK_SIZE) [in HBM]
    """
    n_blocks = x_ref.shape[0]

    # === Pass 1: 扫描全图，计算统计量 (Statistics) ===

    # 初始状态: (max_val, sum_val)
    # 必须显式指定 dtype，否则可能会变成 int
    init_state = (
        jnp.full((1,), -1e9, dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )

    def body_stats(i, carry):
        curr_max, curr_sum = carry

        # [Strict Load] 只有在这里，数据才进入 SRAM
        chunk = x_ref[i]

        # Local Computation
        block_max = jnp.max(chunk)

        # Online Update Logic
        new_max = jnp.maximum(curr_max, block_max)
        # 避免 exp 溢出，虽然理论上 curr_max <= new_max
        rescale = jnp.exp(curr_max - new_max)

        block_sum = jnp.sum(jnp.exp(chunk - new_max))
        new_sum = curr_sum * rescale + block_sum

        # fori_loop 只返回新的 carry，不产生 Output 序列
        return (new_max, new_sum)

    # 使用 fori_loop 替代 scan
    # 语义：从 0 跑到 n_blocks，初始值为 init_state
    final_max, final_sum = jax.lax.fori_loop(0, n_blocks, body_stats, init_state)

    # === Pass 2: 再次扫描，归一化并写回 (Normalize) ===

    # 我们不需要 carry 了，所以 carry 设为 None 或者 dummy 均可
    # 但 fori_loop 要求 body 输入输出结构一致，我们传一个 dummy 进去

    def body_write(i, _):
        # [Strict Load - Second Pass]
        chunk = x_ref[i]

        # Normalize
        # 注意：这里用的是 Pass 1 算出来的最终全局统计量
        val = jnp.exp(chunk - final_max) / final_sum

        # [Strict Store]
        o_ref[i] = val

        return None

    # 再次跑循环
    jax.lax.fori_loop(0, n_blocks, body_write, None)


# %% [markdown]
# ---
#
# ## Part 3: 调度与验证
#
# 这里我们设置 `Grid` 为 Batch 大小。每个 Kernel 实例处理一整行（即处理那行里的所有 Blocks）。[[day07-pallas初探]]
#

# %%
def run_online_softmax(x):
    # x: (Batch, Total_Vocab)
    # 我们先在 Python 端把它 view 成 blocks
    # 假设 x 已经是 (Batch, N_Blocks, Block_Size)
    batch, n_blocks, block_size = x.shape

    # Grid 依然是 (Batch,)，因为我们是一个 Kernel 处理一行
    grid = (batch,)

    # BlockSpec 映射
    # Input: (Batch, N_Blocks, Block_Size)
    # 对于第 i 个任务 (Row i)，我们需要访问 x[i, :, :]
    # 即: 第 0 维随 Grid 变，第 1, 2 维全拿
    spec = pl.BlockSpec(
        index_map=lambda i: (i, 0, 0), block_shape=(1, n_blocks, block_size)
    )

    return pl.pallas_call(
        online_softmax_kernel,  # 使用新的 Kernel
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(spec,),
        out_specs=spec,
    )(x)


# --- 验证 ---
print("Running Online Softmax...")
y_pallas = run_online_softmax(x_sim)

# 还原成 (Batch, Vocab) 进行验证
x_flat = x_sim.reshape(BATCH, -1)
y_flat = y_pallas.reshape(BATCH, -1)

# JAX 标准答案
y_jax = jax.nn.softmax(x_flat, axis=-1)

diff = jnp.max(jnp.abs(y_flat - y_jax))
print(f"Max Diff: {diff:.6f}")
assert diff < 1e-5
print("Success!")

# %% [markdown]
# ---
#
# ## Part 4: 关键总结 (The Bridge to FlashAttention)
#
# ### 为什么这个 Kernel 很重要？
#
# 1.  **SRAM 友好 (O(1) Memory)**：[[day06-计算的物理形态]]
#     即使 `Vocab` 是 100 万，我们的 `chunk` 变量永远只有 `Block_Size` 那么大。SRAM 不会爆炸。
#     我们只保存了两个极小的标量：`run_max` 和 `run_sum`。
#
# 2.  **带宽代价 (The Price)**：
#     我们读了输入 $X$ 两次（Pass 1 读一遍算统计量，Pass 2 读一遍算输出）。
#     写了输出 $O$ 一次。
#     总访存：**2 Read + 1 Write**。
#     这比 Naive Softmax (Read X -> Write Max -> Read X... 4 Read + 3 Write) 好得多，但不如 Element-wise 操作。
