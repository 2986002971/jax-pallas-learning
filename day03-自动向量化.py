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
# 为了演示，我们需要“欺骗”XLA，让它以为我们有 8 张 TPU/GPU。

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
