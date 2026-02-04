# Day 10: 突破内存墙 —— 矩阵乘法与算术强度的博弈

**核心叙事**：
在 Day 09 中，我们分块是为了“能跑通”（解决 SRAM 容量不足）。
在 Day 10 中，我们分块是为了“跑得快”（解决带宽瓶颈）。

矩阵乘法（MatMul）是深度学习的基石。理论上，它的计算量是 $O(N^3)$，数据量是 $O(N^2)$，这应该是一个计算密集型（Compute-Bound）的任务。但如果你写得不好，它瞬间就会变成带宽密集型（Memory-Bound），把强大的 TPU/GPU 饿死。

今天我们要用**算术强度 (Arithmetic Intensity)** 这个物理量，量化证明为什么朴素写法行不通，以及为什么 **Block Strategy** 是唯一的出路。[[day03-自动向量化]]

---

## Part 1: 理论 —— 你的 Kernel 到底在干什么？

**定义**：
$$ \text{Arithmetic Intensity (AI)} = \frac{\text{FLOPs (计算量)}}{\text{Bytes (访存量)}} $$

*   **HBM (显存)**：虽然带宽高达 1TB/s，但相比计算单元还是太慢。
*   **TPU/GPU (算力)**：算力高达几百 TFLOPS。
*   **目标**：为了跑满算力，**AI 必须很高**。如果 AI 很低，计算核心就会停下来等数据（Memory Wall）。

假设我们要计算 $C = A \times B$。我们来看看不同策略下的命运。

### 1.1 朴素点积 (Dot Product) —— 算力的极刑
为了计算 $C$ 的**一个点** $C_{i,j}$：
*   **Read**：读取 $A$ 的第 $i$ 行（$K$ 个数）和 $B$ 的第 $j$ 列（$K$ 个数）。
*   **Compute**：$K$ 次乘法 + $K$ 次加法 = $2K$ FLOPs。
*   **Memory**：读取 $2K$ 个浮点数。假设是 `bfloat16` (2 Bytes)，则是 $4K$ Bytes。

$$ \text{AI}_{\text{naive}} = \frac{2K \text{ FLOPs}}{4K \text{ Bytes}} = 0.5 $$

**结论**：每搬运 1 个 Byte，只做 0.5 次计算。
现代硬件通常需要 AI 达到 **100~300** 才能跑满。**0.5 的强度意味着硬件利用率不到 1%。**

### 1.2 分块 (Tiling) —— 拯救算力的魔法
现在，我们不计算一个点，而是计算 $C$ 的一个 **Block** $(BM, BN)$。
在 Kernel 内部，我们每次从 HBM 搬运一小块 $A$ $(BM, BK)$ 和一小块 $B$ $(BK, BN)$ 到 SRAM。

**Step 1: 搬运 (Load)**
$$ \text{Bytes} = \text{sizeof} \times (BM \cdot BK + BK \cdot BN) $$

**Step 2: 计算 (Compute)**
这两个小块在 SRAM 里做矩阵乘：
$$ \text{FLOPs} = 2 \times BM \times BN \times BK $$

**Step 3: 算术强度推导 (关键)**
$$ \text{AI}_{\text{tile}} = \frac{2 \cdot BM \cdot BN \cdot BK}{\text{sizeof} \cdot (BM \cdot BK + BK \cdot BN)} $$
分子分母都含有 $BK$，约分后（假设 $BM=BN=N_{block}$）：

$$ \text{AI}_{\text{tile}} \approx \frac{1}{\text{sizeof}} \times \frac{2 \cdot N_{block}^2 \cdot BK}{2 \cdot N_{block} \cdot BK} = \frac{N_{block}}{\text{sizeof}} $$

例如，在以下矩阵乘法中，每个矩阵都是 9 块乘以 9 块，我们可以看到，如果我们按行主序计算输出，我们需要将 90 个块加载到 SRAM 中，以计算前 9 个输出块，但如果我们按分组顺序进行，我们只需要加载 54 个块。

![对算术强度的直观理解](figs/grouped_vs_row_major_ordering.png)

### 1.3 深度解读 (The Insight)

这是本章最重要的结论：

1.  **BK 与强度无关**：
    增加 Loop 的步长 $BK$ **不能**提高算术强度。你多读了 $BK$ 的数据，也刚好对应多做了 $BK$ 倍的计算。这两者抵消了。$BK$ 的大小只受限于 SRAM 容量。
2.  **BM, BN 决定生死**：
    **算术强度与输出 Block 的边长成正比**。
    *   Block 边长 1 $\rightarrow$ AI $\approx 0.5$
    *   Block 边长 128 $\rightarrow$ AI $\approx 64$
    这就是为什么我们要用 `128x128` 这样的大块：**为了复用**。

---

## Part 2: 维度规划 (Grid vs Loop)

既然 $BM, BN$ 越大越好，为什么不设成无限大？因为 SRAM 放不下。[[day06-计算的物理形态]]
我们需要在 SRAM 有限的空间内，寻找 $BM, BN$ 的最大值。

**Pallas 的并行策略**：

1.  **Grid 维度 (M, N) —— 空间并行**[[day07-pallas初探]]
    *   我们将输出 $C$ 切分为 Grid。
    *   每个 Kernel 实例负责算一个 $C_{tile}$。
    *   这个 $C_{tile}$ 的大小 $(BM, BN)$ 必须足够大，以保证高 AI。

2.  **Loop 维度 (K) —— 时间累加**
    *   为了算这个 $C_{tile}$，我们需要遍历整个 $K$ 轴。
    *   SRAM 放不下整个 $K$。
    *   所以我们在 Kernel 内部用 `fori_loop`，每次处理 $BK$ 长度的数据。

---

## Part 3: 数据布局 (The 4D View)

为了让 BlockSpec 写得简单，我们延续之前的策略：**在 Python 端预处理**。[[day09-softmax]]

假设 $M=N=1024, K=2048$，Block $(128, 128, 128)$。

*   **Matrix A**: $(M, K) \rightarrow (8, 128, 16, 128)$
    *   为了配合 Grid，我们转置为：`(M_Grid, K_Grid, BM, BK)`
*   **Matrix B**: $(K, N) \rightarrow (16, 128, 8, 128)$
    *   为了配合 Grid，我们转置为：`(K_Grid, N_Grid, BK, BN)`
*   **Matrix C**: $(M, N) \rightarrow (M_Grid, N_Grid, BM, BN)$

---

## Part 4: 代码实现

### 4.1 MatMul Kernel

这个 Kernel 对应 Grid 中的一个点 $(i, j)$。
它的任务是：初始化 Accumulator，遍历 $K_{grid}$，不断搬运 $BK$ 大小的数据块并累加。

```python
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp


def matmul_kernel(a_ref, b_ref, c_ref):
    """
    a_ref: (1, K_Grid, BM, BK) - [Input View] A 的一整行 Block 序列
    b_ref: (K_Grid, 1, BK, BN) - [Input View] B 的一整列 Block 序列
    c_ref: (1, 1, BM, BN)         - [Output View] C 的一个 Tile
    """

    # 1. Accumulator Initialization
    # 这一块内存驻留在 SRAM/Registers 中，读写极快
    acc = jnp.zeros(c_ref.shape, dtype=jnp.float32)

    # 获取 K 维度的步数
    k_steps = b_ref.shape[0]

    # 2. The Inner Loop (K-Dimension Reduction)
    def body(i, current_acc):
        # [Strict Load] 从 HBM 搬运一小块 (Tile) 到 SRAM
        # Pallas 编译器会将其映射为 DMA Copy
        a_tile = a_ref[0, i]  # Shape: (BM, BK)
        b_tile = b_ref[i, 0]  # Shape: (BK, BN)

        # [Compute] 矩阵乘法
        # 这里对应 TPU 的 MXU 或 GPU 的 Tensor Core 指令
        # 此时数据全在 SRAM，速度极快
        return current_acc + jnp.dot(a_tile, b_tile)

    # 执行循环，acc 在循环中不断累加
    acc = jax.lax.fori_loop(0, k_steps, body, acc)

    # 3. [Fusion Opportunity]
    # 如果有 Activation，可以在这里直接做，不消耗带宽
    # acc = jax.nn.relu(acc)

    # 4. [Strict Store] 最终结果写回 HBM
    c_ref[:, :] = acc.astype(c_ref.dtype)
```


### 4.2 Driver & BlockSpec

这是最关键的映射逻辑。我们需要告诉 Pallas：对于 Grid $(i, j)$，去哪里找数据？[[day07-pallas初探]]

*   **A 的逻辑**：Grid $(i, j)$ 需要 $A$ 的第 $i$ 行 block 序列。
    *   Input Shape: `(M_Grid, K_Grid, BM, BK)`
    *   Index Map: `(i, j) -> (i, 0, 0, 0)`
    *   我们锁死第 0 维 ($i$)，并把第 1 维 ($K_{grid}$) **整个**暴露给 Kernel。

*   **B 的逻辑**：Grid $(i, j)$ 需要 $B$ 的第 $j$ 列 block 序列。
    *   Input Shape: `(K_Grid, N_Grid, BK, BN)`
    *   Index Map: `(i, j) -> (0, j, 0, 0)`
    *   我们锁死第 1 维 ($j$)，并把第 0 维 ($K_{grid}$) **整个**暴露给 Kernel。


```python
@jax.jit
def run_pallas_matmul(a, b):
    # a, b: 原始 (M, K), (K, N)
    M, K = a.shape
    K, N = b.shape
    BM, BN, BK = 128, 128, 32
    # BK 的选择：
    # 虽然 BK 只是循环的步长，理论上不影响总数据量。
    # 但如果 BK 过大（如 128），会导致 Shared Memory OOM。
    # 因此，BK 通常取 32 或 64 这种较小的值。

    # --- 1. Reshape for 4D View ---
    # A: (M_Grid, K_Grid, BM, BK)
    a_view = a.reshape(M // BM, BM, K // BK, BK).transpose(0, 2, 1, 3)
    # B: (K_Grid, N_Grid, BK, BN)
    b_view = b.reshape(K // BK, BK, N // BN, BN).transpose(0, 2, 1, 3)
    # 这里我们在 Python 端做了 Reshape 和 Transpose。
    # 优点：让 BlockSpec 的 index_map 写法变得极其简单（直接映射）。
    # 缺点：XLA 可能会在 HBM 中显式创建这些 tensor 的副本，导致显存占用翻倍。
    # 验证：如果你把 K 设得极大（如 50万），会报 HBM OOM，因为 XLA 试图把整个转置矩阵存下来。
    # 真正的 Zero-Copy 优化需要更复杂的 index_map (直接算 stride)，这留给进阶读者探索。

    m_grid = a_view.shape[0]
    n_grid = b_view.shape[1]
    k_grid = a_view.shape[1]  # 或 b_view.shape[0]

    # --- 2. Define BlockSpecs ---

    # Spec A: 选取整行 K Blocks
    in_spec_a = pl.BlockSpec(
        index_map=lambda i, _: (i, 0, 0, 0), block_shape=(1, k_grid, BM, BK)
    )

    # Spec B: 选取整列 K Blocks
    in_spec_b = pl.BlockSpec(
        index_map=lambda _, j: (0, j, 0, 0), block_shape=(k_grid, 1, BK, BN)
    )

    # Spec C: 输出一个 Tile
    out_spec = pl.BlockSpec(
        index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, BM, BN)
    )

    # --- 3. Execute ---
    c_view = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m_grid, n_grid, BM, BN), a.dtype),
        in_specs=(in_spec_a, in_spec_b),
        out_specs=out_spec,
        grid=(m_grid, n_grid),
    )(a_view, b_view)

    # --- 4. Restore Shape ---
    return c_view.transpose(0, 2, 1, 3).reshape(M, N)


# --- Verification ---
k1, k2 = jax.random.split(jax.random.PRNGKey(42))
a = jax.random.normal(k1, (1024, 2048))
b = jax.random.normal(k2, (2048, 1024))

c_pallas = run_pallas_matmul(a, b)
c_jax = jnp.matmul(a, b)

print(f"Max Diff: {jnp.max(jnp.abs(c_pallas - c_jax)):.6f}")
```

---

## Part 5: 总结与伏笔

今天我们通过**算术强度**的理论推导，证明了 $BM, BN$ 的大小是 MatMul 性能的关键。

*   **Grid** 让我们利用了芯片上的所有核心。
*   **Block** 让我们最大化了每次读内存的计算产出。
*   **Loop** 让我们在 SRAM 有限的情况下处理了无限长的 $K$ 维度。

**但是...**
现在的代码还有一个严重的性能隐患：它是**串行**的。
`Load -> Compute -> Load -> Compute`。
当计算单元在全速运转时，内存总线在睡觉；当数据在搬运时，计算单元在休息。

在 **Day 11**，我们将引入 **Pipeline (流水线)** 技术，让 Load 和 Compute 同时进行，进一步榨干硬件性能。
