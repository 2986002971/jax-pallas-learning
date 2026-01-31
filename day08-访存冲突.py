# %% [markdown]
# # Day 8: 极速车道 —— 访存合并与存储体冲突
#
# **核心叙事**：
# 在 **Day 7** 中，我们学会了像厨师切菜一样，把大矩阵切成小块（Tiling）送进 SRAM。你可能觉得：“只要切得足够小，塞进 SRAM，速度就会飞快！”[[day07-pallas初探]]
#
# **现实的打击**：
# 然而，当你写出第一个 Kernel 时，往往会发现它比 JAX 原生算子还要慢。为什么？因为在芯片的世界里，**“怎么拿”（Access Pattern）** 比 **“拿多少”（Tile Size）** 更重要。
#
# 今天，我们要带上显微镜（Profiler），去观察数据在“高速公路”上的行为。
#
# ---
#
# ## Part 1: 理论隐喻 —— 收费站效应
#
# 想象 TPU/GPU 的 **Vector Memory (VMEM)** 不是一个大水桶，而是一排并列的 **8 个收费站（Memory Banks）**。[[day06-计算的物理形态]]
#
# ### 1.1 顺滑模式 (Coalesced Access)
# **场景**：8 辆车排成一横排，同时抵达收费站。
# *   **结果**：车 A 去 1 号窗口，车 B 去 2 号窗口……车 H 去 8 号窗口。
# *   **效率**：所有车在 **1 个时钟周期** 内全部通过。
# *   **代码特征**：读取连续的内存地址，例如 `x[0, 0:8]`。这被称为 **访存合并 (Memory Coalescing)**。
#
# ### 1.2 拥堵模式 (Bank Conflict)
# **场景**：8 辆车虽然同时也到了，但它们都要去 **1 号窗口**（比如都要去 Bank 0）。
# *   **结果**：收费员只能一辆一辆地处理。
# *   **效率**：耗时变成了 **8 个时钟周期**。速度变慢了 8 倍！
# *   **代码特征**：跳跃着读取内存，例如 `x[0:8, 0]`（假设跨度 stride 刚好命中同一个 Bank）。这被称为 **存储体冲突 (Bank Conflict)**。
#
# ---
#
# ## Part 2: 构造案发现场 —— 行规约 vs 列规约
#
# 为了验证这个理论，我们设计两个计算量完全一样的任务，唯一的区别是**读取方向**。
#
# 我们将对一个巨大的 `(8192, 8192)` 矩阵求和。
#
# 1.  **Case A (Row Reduce)**：横着读，顺着加。这是内存最喜欢的模式。[[day03-自动向量化]]
# 2.  **Case B (Col Reduce)**：竖着读，竖着加。这是内存最讨厌的模式。
#
# ### 2.1 准备工作

# %%
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import jax.profiler

# 1. 准备数据
# 8192 x 8192 的 float32 矩阵约 256MB，足够让缓存失效
H, W = 8192, 8192
key = jax.random.PRNGKey(42)
x_input = jax.random.normal(key, (H, W), dtype=jnp.float32)


# %% [markdown]
# ### 2.2 定义 Kernel

# %%
# ==========================================
# Case A: 友好的行规约 (Friendly Row Reduce)
# Block Shape: (8, 1024) -> 主要是横向连续的
# ==========================================
def row_reduce_kernel(x_ref, o_ref):
    # x_ref: (8, 1024)
    # 沿着 axis=1 求和 -> 得到 (8, 1)
    # 读取模式：连续读取 1024 个 float，这利用了 Burst Read
    o_ref[...] = jnp.sum(x_ref[...], axis=1).reshape(o_ref.shape)


# ==========================================
# Case B: 邪恶的列规约 (Hostile Col Reduce)
# Block Shape: (1024, 8) -> 主要是纵向跨步的
# ==========================================
def col_reduce_kernel(x_ref, o_ref):
    # x_ref: (1024, 8)
    # 沿着 axis=0 求和 -> 得到 (1, 8)
    # 读取模式：读 8 个数，跳过 8192 个数...
    # 这会导致极其严重的 Cache Miss 和 Bank Conflict
    o_ref[...] = jnp.sum(x_ref[...], axis=0).reshape(o_ref.shape)


# %% [markdown]
# ---
#
# ## Part 3: 重要的前置知识 —— JIT 与 异步计时
#
# 在开始跑分之前，我们需要补充两个 JAX 的核心知识点，否则你会得到错误的结果。
#
# ### 3.1 加上 `@jax.jit`
# Pallas Kernel 如果直接运行，JAX 会将其视为普通 Python 函数，每次运行都会触发极其昂贵的 **编译 (Compilation)** 过程（PTX -> Cubin）。
# 我们在测速时，只关心 **运行 (Execution)** 时间，不关心编译时间。
# **对策**：必须用 `@jax.jit` 包裹调用函数。[[day02-追踪与编译]]
#
# ### 3.2 使用 `.block_until_ready()`
# JAX 是**异步 (Asynchronous)** 的。当你调用函数时，Python 会立刻返回，而 GPU 还在后台干活。
# 如果你直接用 `time.time()` 计时，你测到的只是“发令枪响”的时间，而不是“跑完比赛”的时间。
# **对策**：在计时结束前，强制同步等待结果。
#
# ### 3.3 封装调用函数
#
# 注意观察 `BlockSpec` 的形状差异。[[day07-pallas初探]]

# %%
@jax.jit
def run_row_reduce_jit(x):
    block_shape = (8, 1024)
    grid = (x.shape[0] // block_shape[0], x.shape[1] // block_shape[1])

    in_spec = pl.BlockSpec(block_shape, lambda i, j: (i, j))
    out_spec = pl.BlockSpec((block_shape[0], 1), lambda i, j: (i, j))

    return pl.pallas_call(
        row_reduce_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (x.shape[0], x.shape[1] // block_shape[1]), x.dtype
        ),
        in_specs=[in_spec],
        out_specs=out_spec,
        grid=grid,
    )(x)


@jax.jit
def run_col_reduce_jit(x):
    block_shape = (1024, 8)
    grid = (x.shape[0] // block_shape[0], x.shape[1] // block_shape[1])

    in_spec = pl.BlockSpec(block_shape, lambda i, j: (i, j))
    out_spec = pl.BlockSpec((1, block_shape[1]), lambda i, j: (i, j))

    return pl.pallas_call(
        col_reduce_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (x.shape[0] // block_shape[0], x.shape[1]), x.dtype
        ),
        in_specs=[in_spec],
        out_specs=out_spec,
        grid=grid,
    )(x)


# %% [markdown]
# ---
#
# ## Part 4: 真相时刻 —— 使用 JAX Profiler
#
# 我们不再瞎猜，而是用专业的工具来“录像”。

# %%
# 1. 预热 (Warmup)
# 这一步至关重要！确保编译完成，不干扰录制
print("Warmup (Compiling)...")
_ = run_row_reduce_jit(x_input).block_until_ready()
_ = run_col_reduce_jit(x_input).block_until_ready()
print("Compilation Done!")

# 2. 开启录制
# trace 会把 GPU 的执行轨迹保存下来
print("Profiling...")
with jax.profiler.trace("./jax-trace-reduce"):
    # 使用 named_scope 给时间轴打标签，方便查找
    with jax.named_scope("Case A: Row Reduce (Fast)"):
        for _ in range(10):
            run_row_reduce_jit(x_input).block_until_ready()

    with jax.named_scope("Case B: Col Reduce (Slow)"):
        for _ in range(10):
            run_col_reduce_jit(x_input).block_until_ready()

print("Done! Trace saved.")

# %% [markdown]
# ### 4.1 如何查看结果
# 1.  找到生成的 trace 文件夹（通常包含 `.json.gz` 文件）。
# 2.  打开浏览器，访问 **[ui.perfetto.dev](https://ui.perfetto.dev/)**。
# 3.  将文件拖入网页。
#
# ### 4.2 读图指南
# 在 Perfetto 中，你会看到两条明显的对比：
#
# *   **Case A (Row Reduce)**：你会看到一排排**短小精悍**的绿色/蓝色条块。这是 GPU 在全速运转。
#     *   *典型耗时*：约 **1.0 ms**。
# *   **Case B (Col Reduce)**：你会看到同样的 Kernel 名字，但条块被**拉长**了，像被稀释了一样。
#     *   *典型耗时*：约 **3.0 ms** 或更长。
#
# **结论**：同样的计算量（都是求和），同样的数据量（都是读 256MB），仅仅因为**访问方向**不同，性能差距达到了 **300%**！
#
# 这多出来的 2ms，就是显存控制器在处理 Bank Conflict 和 DRAM Page Miss 时的“排队等待时间”。
#
# ---
#
# ## Part 5: 总结与思考
#
# 今天我们不仅学到了 Pallas 的性能优化，更重要的是掌握了高性能计算的一条铁律：
#
# 1.  **Memory Coalescing is King**：尽量让每个线程读取相邻的数据。对于 TPU/GPU 来说，**行优先（Row-Major）** 是绝对的舒适区。
# 2.  **Profiling over Guessing**：不要靠感觉优化。使用 `jax.profiler` 配合 `block_until_ready()`，能让你看到微秒级的真相。
# 3.  **Tiling 的陷阱**：在切分 Block 时，不仅要考虑大小，还要考虑形状。`(8, 1024)` 和 `(1024, 8)` 在数学上是对称的，但在物理上是天壤之别。
