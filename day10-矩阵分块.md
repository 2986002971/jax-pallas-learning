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

例如，在以下矩阵乘法中，每个矩阵都是输出 9 个元素，我们可以看到，如果我们按朴素点积计算输出，我们需要将 90 个块加载到 SRAM 中以计算前 9 个输出块，但如果我们按分组顺序进行，我们只需要加载 54 个块。

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

假设 $M=N=1024, K=2048$, 且我们选择 $BM=BN=128, BK=32$。

*   **Matrix A**: $(M, K) \rightarrow (M_{grid}, BM, K_{grid}, BK)$
    *   Shape: `(8, 128, 64, 32)`
*   **Matrix B**: $(K, N) \rightarrow (K_{grid}, BK, N_{grid}, BN)$
    *   Shape: `(64, 32, 8, 128)`
*   **Matrix C**: $(M, N) \rightarrow (M_{grid}, BM, N_{grid}, BN)$
    *   Shape: `(8, 128, 8, 128)`
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
    a_ref: (1, BM, K_Grid, BK)
    b_ref: (K_Grid, BK, 1, BN)
    c_ref: (1, BM, 1, BN)
    """

    acc = jnp.zeros(c_ref.shape, dtype=jnp.float32).squeeze((0, 2))  # (BM, BN)

    # 这里的 K_Grid 位于 b_ref 的第 0 维
    k_steps = b_ref.shape[0]

    def body(k, current_acc):
        # A: 取第 k 个 block。维度索引：(0, all_rows, k, all_cols)
        a_tile = a_ref[0, :, k, :]

        # B: 取第 k 个 block。维度索引：(k, all_rows, 0, all_cols)
        b_tile = b_ref[k, :, 0, :]

        return current_acc + jnp.dot(a_tile, b_tile)

    acc = jax.lax.fori_loop(0, k_steps, body, acc)

    # Output: (1, BM, 1, BN) -> 需要写回 (BM, BN)
    c_ref[0, :, 0, :] = acc.astype(c_ref.dtype)
```


### 4.2 Driver & BlockSpec

这是最关键的映射逻辑。我们需要告诉 Pallas：对于 Grid $(i, j)$，去哪里找数据？[[day07-pallas初探]]

*   **A 的逻辑**：Grid $(i, j)$ 需要 $A$ 的第 $i$ 行 block 序列。
    *   Input View: `(M_Grid, BM, K_Grid, BK)`
    *   Index Map: `(i, j) -> (i, 0, 0, 0)`
    *   我们锁死第 0 维 ($i$)，并把第 2 维 ($K_{grid}$) **整个**暴露给 Kernel。
    *   Kernel 看到的 Shape: `(1, BM, K_Grid, BK)`

*   **B 的逻辑**：Grid $(i, j)$ 需要 $B$ 的第 $j$ 列 block 序列。
    *   Input View: `(K_Grid, BK, N_Grid, BN)`
    *   Index Map: `(i, j) -> (0, 0, j, 0)`
    *   我们锁死第 2 维 ($j$)，并把第 0 维 ($K_{grid}$) **整个**暴露给 Kernel。
    *   Kernel 看到的 Shape: `(K_Grid, BK, 1, BN)`



```python
@jax.jit
def run_pallas_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    BM, BN, BK = 128, 128, 32
    # BK 的选择：
    # 虽然 BK 只是循环的步长，理论上不影响算术强度。
    # 但如果 BK 过大（如 128），会导致 Shared Memory OOM。
    # 因此，BK 通常取 32 或 64 这种较小的值。

    # --- 1. Reshape Only (Zero Copy) ---
    # Reshape 操作只是改变了数据的视图 (View)，我们只改变看待数据的视角，不移动任何数据
    # A: (M_Grid, BM, K_Grid, BK)
    a_view = a.reshape(M // BM, BM, K // BK, BK)

    # B: (K_Grid, BK, N_Grid, BN)
    b_view = b.reshape(K // BK, BK, N // BN, BN)

    m_grid = a_view.shape[0]
    n_grid = b_view.shape[2]
    k_grid = b_view.shape[0]

    # --- 2. Define BlockSpecs ---

    # Spec A:
    # Input: (M_Grid, BM, K_Grid, BK)
    # 我们要锁定 M_Grid=i, 取所有的 BM, K_Grid, BK
    # Mapping: (i, j) -> (i, 0, 0, 0)
    # Shape: (1, BM, K_Grid, BK)
    in_spec_a = pl.BlockSpec(
        index_map=lambda i, _: (i, 0, 0, 0), block_shape=(1, BM, k_grid, BK)
    )

    # Spec B:
    # Input: (K_Grid, BK, N_Grid, BN)
    # 我们要锁定 N_Grid=j, 取所有的 K_Grid, BK, BN
    # Mapping: (i, j) -> (0, 0, j, 0)
    # Shape: (K_Grid, BK, 1, BN)
    in_spec_b = pl.BlockSpec(
        index_map=lambda _, j: (0, 0, j, 0), block_shape=(k_grid, BK, 1, BN)
    )

    # Spec C:
    # Output: (M_Grid, BM, N_Grid, BN)
    out_spec = pl.BlockSpec(
        index_map=lambda i, j: (i, 0, j, 0), block_shape=(1, BM, 1, BN)
    )

    # --- 3. Execute ---
    # Output Shape 也要对应 Reshape 后的布局
    c_view = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m_grid, BM, n_grid, BN), a.dtype),
        in_specs=(in_spec_a, in_spec_b),
        out_specs=out_spec,
        grid=(m_grid, n_grid),
    )(a_view, b_view)

    # --- 4. Restore Shape ---
    # C 是 (M_Grid, BM, N_Grid, BN) -> reshape -> (M, N)
    return c_view.reshape(M, N)


# --- Verification ---
k1, k2 = jax.random.split(jax.random.PRNGKey(42))
a = jax.random.normal(k1, (1024, 2048))
b = jax.random.normal(k2, (2048, 1024))

c_pallas = run_pallas_matmul(a, b)
c_jax = jnp.matmul(a, b)

print(f"Max Diff: {jnp.max(jnp.abs(c_pallas - c_jax)):.6f}")
```


---

## Part 5: 实验验证 —— Pallas 的内存行为大揭秘

在 Part 4 的代码中，我们留下了一个悬念：我们将 `BlockSpec` 定义为包含**整个 $K$ 维度**。
这意味着，逻辑上不仅 $BM, BN$ 在 Block 里，连巨大的 $K$ 也在 Block 的定义里。

**灵魂拷问**：
> 既然 Block 对应 SRAM，而 SRAM 只有几百 KB。为什么当我们把 $K$ 设为几万（几百 MB 数据）时，SRAM 没有被撑爆？

为了回答这个问题，我们设计了一组**“双盲压力测试”**。

### 5.1 实验 A：挑战显存 (HBM) 极限
**测试逻辑**：保持 `BK=32` 不变（即每次循环只读 32 个数），疯狂增加总长度 $K$。


```python
def stress_test_hbm_limit():
    print("\n--- Experiment A: Stressing HBM (Increasing Total K) ---")
    M, N = 1024, 1024

    current_k = 65536  # 起步就是 6.5万

    while True:
        try:
            print(f"Testing K={current_k}...", end=" ")
            # 只分配数据，尚未触发 Kernel
            a = jax.random.normal(jax.random.PRNGKey(0), (M, current_k))
            b = jax.random.normal(jax.random.PRNGKey(1), (current_k, N))

            # 运行 Kernel
            run_pallas_matmul(a, b).block_until_ready()
            print(
                f"Success! (Total Matrix Size: {a.nbytes / 1e9 + b.nbytes / 1e9:.2f} GB)"
            )

            current_k *= 2  # 压力翻倍

        except Exception as e:
            print(f"\n[FAILED] at K={current_k}")
            if "RESOURCE_EXHAUSTED" in str(e) or "allocator" in str(e):
                print(">> 错误类型: Global Memory (HBM) OOM")
                print(">> 结论: 数据量太大，HBM装不下了。但 SRAM 没爆。")
            else:
                print(f">> 错误: {e}")
            break


stress_test_hbm_limit()
```


**实验结果与日志分析**：
直到 $K$ 增加到几十万，导致单张矩阵大小达到数GB时，程序崩溃：
```text
[FAILED] at K=524288
>> 错误: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.00GiB.
... Allocator (GPU_0_bfc) ran out of memory ...
```

**解读**：
1.  **凶手是 `GPU_0_bfc`**：这是 JAX 管理 **HBM (显存)** 的分配器。说明是显存条上的空间不够用了（或者碎片化太严重，找不到连续的 2GB 空间）。
2.  **SRAM 的不在场证明**：如果 BlockSpec 真的试图把这 2GB 数据塞进 SRAM，早在 $K$ 刚开始增加时，就已经触发 `Shared memory size limit` 了。
3.  **结论**：`Ref` 在 $K$ 维度是**流式 (Streaming)** 的。虽然 BlockSpec 写了整个 $K$，但实际进入 SRAM 的只有循环当前的 $BK$。

### 5.2 实验 B：挑战片上内存 (SRAM) 极限
**测试逻辑**：保持总长度 $K$ 不变，疯狂增加 `BK`（单次循环读取的数据量）。
理论上，如果 `Ref` 的数据是驻留在 SRAM 里的，那么稍微增加 `BK` 就会立即触碰物理天花板。


```python
def stress_test_sram_limit():
    print("\n--- Experiment B: Stressing SRAM (Increasing Tile Size BK) ---")
    M, N, K = 1024, 1024, 4096  # 固定一个较小的 K
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    a = jax.random.normal(k1, (M, K))
    b = jax.random.normal(k2, (K, N))

    def run_pallas_matmul_dynamic_bk(a, b, bk=None):
        M, K = a.shape
        K, N = b.shape
        BM, BN = 128, 128
        BK = bk if bk is not None else 32

        a_view = a.reshape(M // BM, BM, K // BK, BK)
        b_view = b.reshape(K // BK, BK, N // BN, BN)

        m_grid = a_view.shape[0]
        n_grid = b_view.shape[2]
        k_grid = b_view.shape[0]

        in_spec_a = pl.BlockSpec(
            index_map=lambda i, _: (i, 0, 0, 0), block_shape=(1, BM, k_grid, BK)
        )
        in_spec_b = pl.BlockSpec(
            index_map=lambda _, j: (0, 0, j, 0), block_shape=(k_grid, BK, 1, BN)
        )
        out_spec = pl.BlockSpec(
            index_map=lambda i, j: (i, 0, j, 0), block_shape=(1, BM, 1, BN)
        )

        c_view = pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((m_grid, BM, n_grid, BN), a.dtype),
            in_specs=(in_spec_a, in_spec_b),
            out_specs=out_spec,
            grid=(m_grid, n_grid),
        )(a_view, b_view)

        return c_view.reshape(M, N)

    # 逐渐增大 BK
    for bk_test in [16, 32, 64, 128, 256, 512, 1024]:
        try:
            print(f"Testing BK={bk_test}...", end=" ")

            run_pallas_matmul_dynamic_bk(a, b, bk=bk_test)

            # 估算 SRAM 占用: 两个输入块 (BM*BK + BK*BN) + 一个累加器 (BM*BN)
            # float32 = 4 bytes
            sram_usage = (128 * bk_test + bk_test * 128 + 128 * 128) * 4 / 1024
            print(f"Success! (Est. SRAM usage: {sram_usage:.1f} KB)")

        except Exception as e:
            print(f"\n[FAILED] at BK={bk_test}")
            # Pallas/XLA 通常会报 Shared Memory 相关的错误
            print(f">> 错误信息摘要: {str(e)[:200]}...")
            print(">> 结论: Tile 太大，SRAM (Shared Memory) 立即爆炸。")
            break


stress_test_sram_limit()
```

**实验结果**：
当 `BK` 增加到 64 时，立即触发了报错：
```text
Testing BK=32... Success! (Est. SRAM usage: 96.0 KB)
Testing BK=64...
[FAILED] at BK=64
>> 错误信息摘要: RESOURCE_EXHAUSTED: Shared memory size limit exceeded: requested 131072, available: 101376...
```

**解读**：
1.  **精准的阈值**：`BK=64` 时，仅输入数据就会占用 $128 \times 64 \times 4 \text{Bytes} \times 2 \approx 64 \text{KB}$，加上输出 block 和其他开销，瞬间突破了该 GPU 具体的 Shared Memory 限制（100KB 左右）。
2.  **结论**：**Block 维度是驻留 (Resident) 的**。你定义了多大的 Block，就必须有这么大的 SRAM。

### 5.3 结论

这两个实验如同两块拼图，拼出了 Pallas 内存管理的完整图景：

1.  **BlockSpec 的“谎言”**：
    当我们定义 `block_shape` 时，Pallas 并没有把所有定义的数据都塞进 SRAM。它不仅看 Shape，还看你在 Kernel 里**怎么用**。

2.  **Kernel 的“真相”**：
    *   **Grid 维度 (M, N, K_grid)**：存在于 **HBM**。访问它们就像翻书，翻到哪一页，哪一页才会被加载。
    *   **Loop 步长 (BK)**：存在于 **SRAM**。这是你眼睛一次能看到的页面大小。必须物理上容纳得下，否则直接报错。

这也解释了为什么我们在 Part 4 的代码是安全的，但也指出了优化的方向：既然 $K$ 是流式的，我们是否可以利用这个特性进行 **Pipeline（流水线）** 优化，掩盖读取 $BK$ 的时间？

---

## Part 6: 总结与伏笔

今天我们通过**算术强度**的理论推导，证明了 $BM, BN$ 的大小是 MatMul 性能的关键。

*   **Grid** 让我们利用了芯片上的所有核心。
*   **Block** 让我们最大化了每次读内存的计算产出。
*   **Loop** 让我们在 SRAM 有限的情况下处理了无限长的 $K$ 维度。

**但是...**
现在的代码还有一个严重的性能隐患：它是**串行**的。
`Load -> Compute -> Load -> Compute`。
当计算单元在全速运转时，内存总线在睡觉；当数据在搬运时，计算单元在休息。

在 **Day 11**，我们将引入 **Pipeline (流水线)** 技术，让 Load 和 Compute 同时进行，进一步榨干硬件性能。
