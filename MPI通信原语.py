# %% [markdown]
# # Day 11: 突破互联瓶颈 —— 环形通信的艺术
#
# **核心叙事**：
# 当我们将模型扩展到 1000 张 GPU 时，“计算”往往只要几毫秒，而“互相通气”却要花几百毫秒。如果通信策略不当，算力再强也得停下来等数据。
#
# 今天，我们将深入分布式系统的血管——**集合通信 (Collective Communication)**。我们将溯源至 **MPI 标准**，拆解经典的 **Ring AllReduce** 算法，最后通过 **JAX HLO** 的显微镜审视，看看现代深度学习编译器是如何自动编排这些底层动作的。
#
# ---
#
# ## Part 1: 上古卷轴 —— 从 MPI 到 AllReduce
#
# 在深度学习爆发前的几十年，高性能计算（HPC）领域的物理学家和气象学家们就已经在使用超级计算机了。他们定义了一套标准叫 **MPI (Message Passing Interface)**。
#
# 在分布式训练中，我们最常见的需求是：**每张卡都算出了一部分梯度，如何让所有人都能得到“所有梯度的总和”？**
#
# ### 1.1 朴素架构的崩溃：Parameter Server
#
# 最直观的想法是选一个“班长”（Parameter Server/Master Node）。
# 1.  **工人们**（Workers）把自己算好的梯度发给班长。
# 2.  **班长** 把收到的梯度加起来。
# 3.  **班长** 把总和发回给所有工人。
#
# $$ \text{时间} \propto \text{工人数量 } N \times \text{数据量 } M $$
#
# *   **瓶颈**：班长的网卡会瞬间被挤爆。随着 GPU 数量增加，网络拥堵呈线性爆炸。这在千卡集群上是不可接受的。
#
# ### 1.2 环形架构（Ring）：民主的胜利
#
# 为了解决单点瓶颈，百度（没错，是 Baidu Research 在 2017 年把这个 HPC 技术引入了 DL）提出了 **Ring AllReduce**。
#
# 它的核心思想是：**没有班长，人人都是班长。数据像接力棒一样在环上传递。**
#
# 神奇的结论是：**Ring 算法的通信时间几乎与 GPU 数量无关！**
# 无论你有 8 张卡还是 8000 张卡，每个节点只需要发送 $2M$ 的数据量。
#
# 为了理解这个“反直觉”的结论，我们需要把 AllReduce 拆解为两个更基础的 MPI 原语：**ReduceScatter** + **AllGather**。
#
# ---
#
# ## Part 2: 拆解第一阶段 —— ReduceScatter (归约散播)
#
# ### 0. 设定场景
#
# *   **节点（Rank）：** 0, 1, 2。
# *   **拓扑结构（Ring）：** $0 \to 1 \to 2 \to 0$ （单向环）。
# *   **总步数：** $N-1 = 2$ 步。
# *   **数据切分：** 假设每个节点都有一个向量，被切分成 3 块：
#     *   🟥 **第 0 块 (Red):** 最终要在 Rank 0 汇总。
#     *   🟩 **第 1 块 (Green):** 最终要在 Rank 1 汇总。
#     *   🟦 **第 2 块 (Blue):** 最终要在 Rank 2 汇总。
#
# **初始状态（Step 0）：每个人手里都有全部数据的“原始版本”**
#
# | 节点 | 缓冲区内容 (从左到右: 第0块, 第1块, 第2块) |
# | :--- | :--- |
# | **Rank 0** | [ 🟥 $d_{0,0}$ , 🟩 $d_{0,1}$ , 🟦 $d_{0,2}$ ] |
# | **Rank 1** | [ 🟥 $d_{1,0}$ , 🟩 $d_{1,1}$ , 🟦 $d_{1,2}$ ] |
# | **Rank 2** | [ 🟥 $d_{2,0}$ , 🟩 $d_{2,1}$ , 🟦 $d_{2,2}$ ] |
#
# ---
#
# ### 1. 第一步 (Step 1)
#
# **策略：** 每个节点发送“**目标是下下家**”的那块数据给下家。为了方便记忆，我们关注**谁要把数据发出去**。
# *   第 2 块的目标是 Rank 2 -> 所以 Rank 0 把第 2 块发给 Rank 1（让 Rank 1 帮忙累加）。
# *   第 0 块的目标是 Rank 0 -> 所以 Rank 1 把第 0 块发给 Rank 2（让 Rank 2 帮忙累加）。
# *   第 1 块的目标是 Rank 1 -> 所以 Rank 2 把第 1 块发给 Rank 0（让 Rank 0 帮忙累加）。
#
# **动作：**
# *   **Rank 0** 发送 🟦 $d_{0,2}$ $\to$ **Rank 1**
# *   **Rank 1** 发送 🟥 $d_{1,0}$ $\to$ **Rank 2**
# *   **Rank 2** 发送 🟩 $d_{2,1}$ $\to$ **Rank 0**
#
# **Step 1 结束后的状态（接收并累加）：**
#
# | 节点 | 缓冲区发生的变化 | 当前持有的关键“半成品” |
# | :--- | :--- | :--- |
# | **Rank 0** | 收到 Rank 2 发来的 🟩 $d_{2,1}$ | 🟩 $tmp_1 = d_{0,1} + d_{2,1}$ <br>(Rank 0 手里有了 0和2 的绿色和) |
# | **Rank 1** | 收到 Rank 0 发来的 🟦 $d_{0,2}$ | 🟦 $tmp_2 = d_{1,2} + d_{0,2}$ <br>(Rank 1 手里有了 1和0 的蓝色和) |
# | **Rank 2** | 收到 Rank 1 发来的 🟥 $d_{1,0}$ | 🟥 $tmp_0 = d_{2,0} + d_{1,0}$ <br>(Rank 2 手里有了 2和1 的红色和) |
#
# ---
#
# ### 2. 第二步 (Step 2) —— 最后一步
#
# **策略：** 继续接力。把刚才算好的“半成品”传给下家。这次传给下家后，刚好到达该数据块的目标节点。
#
# **动作：**
# *   **Rank 0** 发送刚才算好的 🟩 $tmp_1$ $\to$ **Rank 1** (Rank 1 是绿块的终点)
# *   **Rank 1** 发送刚才算好的 🟦 $tmp_2$ $\to$ **Rank 2** (Rank 2 是蓝块的终点)
# *   **Rank 2** 发送刚才算好的 🟥 $tmp_0$ $\to$ **Rank 0** (Rank 0 是红块的终点)
#
# **Step 2 结束后的状态（接收并累加）：**
#
# | 节点 | 缓冲区发生的变化 |结果 |
# | :--- | :--- | :--- |
# | **Rank 0** | 收到 Rank 2 发来的 🟥 $tmp_0$ | 🟥 **最终结果** = $d_{0,0} + tmp_0 = d_{0,0} + d_{2,0} + d_{1,0}$ <br> (Rank 0 拿到了完整的红色块) |
# | **Rank 1** | 收到 Rank 0 发来的 🟩 $tmp_1$ | 🟩 **最终结果** = $d_{1,1} + tmp_1 = d_{1,1} + d_{0,1} + d_{2,1}$ <br> (Rank 1 拿到了完整的绿色块) |
# | **Rank 2** | 收到 Rank 1 发来的 🟦 $tmp_2$ | 🟦 **最终结果** = $d_{2,2} + tmp_2 = d_{2,2} + d_{1,2} + d_{0,2}$ <br> (Rank 2 拿到了完整的蓝色块) |
#
# ---
#
# ### 3. 结果验证
#
# ReduceScatter 结束后：
# *   **Rank 0** 此时只关心它的 `recvbuf`，里面装着完美的 $\sum$ **第 0 块** (🟥)。
# *   **Rank 1** 此时只关心它的 `recvbuf`，里面装着完美的 $\sum$ **第 1 块** (🟩)。
# *   **Rank 2** 此时只关心它的 `recvbuf`，里面装着完美的 $\sum$ **第 2 块** (🟦)。
#
# 至于其他块的数据（比如 Rank 0 手里剩下的那一半蓝块和绿块），它们已经在计算过程中被“贡献”出去了，留下的只是残缺的半成品或旧数据，不再被需要，通常会被这里的结果覆盖或者被丢弃。
#
# 这就是为什么叫 **Scatter (散播)**：完整的结果被拆散了，没人拥有全部结果，每人手里只攥着属于自己的那一小块最终拼图。
#
# ---
#
# ## Part 3: 拆解第二阶段 —— AllGather (全收集)
#
# 好的，这种带颜色方块和“手牌”表格的展示方式确实不仅直观，而且能清晰地看到数据是如何在环上流动的。
#
# 我将仿照你提供的 **ReduceScatter** 的风格，重新拆解 **Part 3: AllGather (全收集)** 的过程。
#
# ---
#
# ## Part 3: 拆解第二阶段 —— AllGather (全收集)
#
# **核心逻辑差异：**
# *   **ReduceScatter** 是“边跑边加”，数据在传的过程中不断融合，最后每人只拿一块结果。
# *   **AllGather** 是“边跑边存”，数据在传的过程中保持原样，纯粹是互相交换，最后每人把所有块都拼齐。
#
# 我们沿用刚才的设定：**3 个节点 (N=3)**，**Rank 0, 1, 2**，环形拓扑 $0 \to 1 \to 2 \to 0$。
#
# ### 0. 初始状态 (场景设定)
#
# 注意：AllGather 的初始状态通常是接着 ReduceScatter 之后的。
# 这意味着：**每个人手里只有属于自己的那一块完整数据（完美求和后的结果）**，其他位置是空的。
#
# | 节点 | 手里的数据 (从左到右: 第0块, 第1块, 第2块) | 状态描述 |
# | :--- | :--- | :--- |
# | **Rank 0** | [ 🟥 **Full**, `--`, `--` ] | (Rank 0 负责红块，此时它只有红块是全的) |
# | **Rank 1** | [ `--`, 🟩 **Full**, `--` ] | (Rank 1 负责绿块，此时它只有绿块是全的) |
# | **Rank 2** | [ `--`, `--`, 🟦 **Full** ] | (Rank 2 负责蓝块，此时它只有蓝块是全的) |
#
# **目标：** 填满表格里的空缺 (`--`)，让每个人都集齐 [ 🟥, 🟩, 🟦 ]。
# **总步数：** 同样是 $N-1 = 2$ 步。
#
# ---
#
# ### 1. 第一步 (Step 1)
#
# **策略：** 把你手里**刚拿到的、最新鲜的**那块数据，传给你的下家。
# *(第一步时，“最新鲜的”就是你自己本来拥有的那块核心资产)*。
#
# **动作：**
# *   **Rank 0** 将手中的 🟥 **Full(第0块)** $\to$ 发给 **Rank 1** (Rank 1 缺红块)
# *   **Rank 1** 将手中的 🟩 **Full(第1块)** $\to$ 发给 **Rank 2** (Rank 2 缺绿块)
# *   **Rank 2** 将手中的 🟦 **Full(第2块)** $\to$ 发给 **Rank 0** (Rank 0 缺蓝块)
#
# **Step 1 结束后的状态（接收并 `Copy` 填空）：**
#
# | 节点 | 缓冲区变化 | 当前手里的牌 |
# | :--- | :--- | :--- |
# | **Rank 0** | 收到 Rank 2 发来的 🟦 | [ 🟥, `--`, 🟦 ] <br>(Rank 0 刚才缺蓝块，现在补上了) |
# | **Rank 1** | 收到 Rank 0 发来的 🟥 | [ 🟥, 🟩, `--` ] <br>(Rank 1 刚才缺红块，现在补上了) |
# | **Rank 2** | 收到 Rank 1 发来的 🟩 | [ `--`, 🟩, 🟦 ] <br>(Rank 2 刚才缺绿块，现在补上了) |
#
# > **观察：** 此时大家都完成了 2/3 的拼图。
#
# ---
#
# ### 2. 第二步 (Step 2) —— 最后一步
#
# **策略：** 继续接力。**把你上一轮刚收到的那块数据**，原封不动地传给你的下家。
# *(为什么？以 Rank 0 为例，它刚从 Rank 2 收到了蓝块。Rank 1 还没收到蓝块呢，Rank 0 需要充当搬运工，把蓝块继续传给 Rank 1)*。
#
# **动作：**
# *   **Rank 0** 将(上一步刚收到的) 🟦 **Full(第2块)** $\to$ 发给 **Rank 1**
# *   **Rank 1** 将(上一步刚收到的) 🟥 **Full(第0块)** $\to$ 发给 **Rank 2**
# *   **Rank 2** 将(上一步刚收到的) 🟩 **Full(第1块)** $\to$ 发给 **Rank 0**
#
# **Step 2 结束后的状态（接收并 `Copy` 填空）：**
#
# | 节点 | 缓冲区变化 | 最终结果 |
# | :--- | :--- | :--- |
# | **Rank 0** | 收到 Rank 2 发来的 🟩 | [ 🟥, 🟩, 🟦 ] **集齐！** |
# | **Rank 1** | 收到 Rank 0 发来的 🟦 | [ 🟥, 🟩, 🟦 ] **集齐！** |
# | **Rank 2** | 收到 Rank 1 发来的 🟥 | [ 🟥, 🟩, 🟦 ] **集齐！** |
#
# ---
#
# ### 3. 数据流转总结
#
# 如果我们追踪每一块颜色的轨迹，会发现它们各自走完了一个环，只是起点不同：
#
# *   **🟥 第 0 块 (红)** : Rank 0 (初始) $\to$ Rank 1 (第1步收到) $\to$ Rank 2 (第2步收到)。
# *   **🟩 第 1 块 (绿)** : Rank 1 (初始) $\to$ Rank 2 (第1步收到) $\to$ Rank 0 (第2步收到)。
# *   **🟦 第 2 块 (蓝)** : Rank 2 (初始) $\to$ Rank 0 (第1步收到) $\to$ Rank 1 (第2步收到)。
#
# 跑完 $N-1$ 步后，任何一块数据都访问了环上的每一个节点。
#
# ---
#
# ### 4. 最终大成：Ring AllReduce
#
# 现在你把 Part 2 和 Part 3 连起来看，就是完整的 **Ring AllReduce** 流程：
#
# 1.  **ReduceScatter 阶段** (散播求和)：
#     *   大家把手里的数据一点点加起来，最后化整为零。
#     *   **结局**：Rank 0 独占求和后的红块，Rank 1 独占绿块，Rank 2 独占蓝块。
#     *   **通信量**：每个节点发送了 $\frac{N-1}{N} \times M$ 大小的数据。
#
# 2.  **AllGather 阶段** (全收集)：
#     *   大家把手里独占的那块精华广播出去。
#     *   **结局**：所有人都拥有了完整的 [红+绿+蓝] 总和。
#     *   **通信量**：每个节点又发送了 $\frac{N-1}{N} \times M$ 大小的数据。
#
# **总通信量公式验证：**
# $$ \text{Total Data Sent} = \text{ReduceScatter} + \text{AllGather} = 2 \times \frac{N-1}{N} \times M $$
# 当 $N$ (显卡数量) 很大时，$\frac{N-1}{N}$ 约等于 1。
# 所以总通信量恒定为 $2M$，与显卡数量无关！这就是 Ring AllReduce 能够支持大规模分布式训练的核心秘密。
#
# ---
#
# ## Part 4: 现代视角 —— JAX 下的“隐式通信”
#
# 在 PyTorch DDP 时代，你可能需要手动写 `dist.all_reduce`。但在 JAX 中，我们进入了 **SPMD (Single Program Multiple Data)** 时代。
#
# 你只管写矩阵乘法，XLA 编译器会自动发现“诶，数据被切开了，这里需要插入一个 AllReduce”。
#
# 让我们用代码来验证这一点。
#
# ### 4.1 搭建舞台：欺骗 XLA (8 卡模拟)
#
# 我们没有 8 张 A100，但我们可以通过环境变量欺骗 XLA，把 CPU 切成 8 份。这足以骗过编译器生成分布式代码。

# %%
import os

# 伪装 8 个设备 (必须在 import jax 之前设置)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.update("jax_platform_name", "cpu")

# 创建 2x4 的逻辑网格
# 'data': 用于数据并行 (Batch Dim) -> 2份
# 'model': 用于张量并行 (Hidden Dim) -> 4份
devices = mesh_utils.create_device_mesh((2, 4))
mesh = Mesh(devices, axis_names=("data", "model"))
print(f"虚拟设备网格: {mesh}")

# %% [markdown]
# ### 4.2 定义切分意图：PartitionSpec
#
# 我们要模拟一个典型的**张量并行 (Tensor Parallelism)** 场景。
# 假设都在做 $Y = X @ W$。
#
# *   $X$ (输入): 我们切分它的**列 (Feature Dim)**。
# *   $W$ (权重): 我们切分它的**行 (Input Feature Dim)**。
#
# 注意：$X$ 的列和 $W$ 的行是相乘并累加的维度（收缩维）。**如果你切分了这个维度，意味着每张卡只能计算出“部分和”，必须通信才能得到最终结果。**

# %%
# 数据维度
B, Seq, Hidden, Out = 4, 128, 16384, 16384

# 定义切分策略
# X: [Batch, Seq, Hidden]
# 我们把 Hidden 维切到了 'model' 轴 (4份)
spec_X = P("data", None, "model")

# W: [Hidden, Out]
# 这里的 Hidden 维也必须对其切分到 'model' 轴
spec_W = P("model", None)

# Y: [Batch, Seq, Out]
# 结果里 Hidden 维消失了。理论上它是全量的。
spec_Y = P("data", None, None)


# %% [markdown]
# ### 4.3 HLO 代码分析
#
# 现在我们定义计算函数，并让 JAX 编译它。我们将深入 **StableHLO** 代码，寻找编译器插入通信算子的铁证。
#
#

# %%
# 1. 定义计算图
@jax.jit(
    in_shardings=(NamedSharding(mesh, spec_X), NamedSharding(mesh, spec_W)),
    out_shardings=NamedSharding(mesh, spec_Y),
)
def parallel_matmul(x, w):
    return jnp.dot(x, w)


# 2. 构造 Dummy Data
key = jax.random.PRNGKey(0)
x_dummy = jax.random.normal(key, (B, Seq, Hidden))
w_dummy = jax.random.normal(key, (Hidden, Out))

# 3. 注入分片信息 (Compile with Sharding)
lowered = parallel_matmul.lower(x_dummy, w_dummy)

# 4. 获取编译后的 HLO 代码
hlo_text = lowered.compile().as_text()

print("\n=== HLO 通信指令侦测 ===")

# 在 HLO 中寻找 collective ops
# 注意：在 StableHLO 中，all-reduce 通常叫 'all-reduce'
# 参数中会包含 replica_groups (描述哪些卡在一起通信)
ops_to_find = ["all-reduce", "reduce-scatter", "all-gather"]
found_ops = []

for line in hlo_text.splitlines():
    for op in ops_to_find:
        if op in line and op not in found_ops:
            found_ops.append(op)
            print(f"✅ 发现指令: {line.strip()[:80]}...")

if not found_ops:
    print("❌ 未发现通信指令。请检查切分策略。")
else:
    print("\n[分析结果]:")
    print("XLA 检测到你在 'model' 轴切分了收缩维(Contracting Dim)。")
    print("为了保证数学正确性，它自动插入了 collective ops 来聚合各卡的部分和。")

# %% [markdown]
# ### 4.4 实验分析
#
# 如果你运行这段代码，你应该会看到类似这样的输出：
#
# ```text
# ✅ 发现指令: %all-reduce = f32[4,128,512]{2,1,0} all-reduce(...)...
# ```
#
# **解读**：
# 1.  **`all-reduce`**: 果然！编译器发现了需要聚合。
# 2.  **`replica_groups={{0,1,2,3},{4,5,6,7}}`**: 这非常有意思。我们在定义 `mesh` 时，`model` 轴大小是 4。
#     *   设备 0,1,2,3 组成了一个通信组（Ring）。
#     *   设备 4,5,6,7 组成了另一个通信组。
#     *   这说明 AllReduce 只发生在 `model` 轴内部，这正是**张量并行**的特征（只在模型并行的组内同步，不需要跨数据并行组同步）。
#
# ---
#
# ## Part 5: 总结
#
# 今天我们完成了一次从理论到实现的跨越：
#
# 1.  **MPI 的智慧**：我们不需要 Parameter Server 这种瓶颈。通过 **Ring (环形)** 算法，我们将大块数据的聚合拆解为 **ReduceScatter + AllGather**，实现了与节点数无关的高效通信。
# 2.  **JAX 的哲学**：你不需要显式地编写 `Ring` 代码。你只需要通过 `PartitionSpec` 告诉编译器“数据在哪里”，XLA 就会像一个经验丰富的 HPC 工程师，自动为你插入最高效的 `all-reduce` 指令。
#
# 这就像你要寄快递。
# 在 MPI 时代，你需要自己根据地图规划卡车路线（Ring Algorithm）。
# 在 JAX 时代，你只需要填写“收件地址”（PartitionSpec），顺丰（XLA）会自动帮你搞定所有的物流调度。
