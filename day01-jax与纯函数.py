# %% [markdown]
# ### Part 1: PyTree —— JAX 的通用数据容器
#
# 在 PyTorch 中，我们习惯于 `model(x)`，但在 JAX 中，我们经常看到 `apply_fn(params, x)`。这里的 `params` 通常是一个复杂的嵌套字典，这就是 PyTree 的一种。
#
# #### 1. 什么是 PyTree？
# 简单来说，**PyTree 是 Python 容器结构的抽象**。它是一个递归定义的概念：
# *   **叶子节点 (Leaves)**：各种数组（JAX arrays, NumPy arrays）、标量、或者 `None`。
# *   **节点 (Nodes)**：由列表 (list)、元组 (tuple)、字典 (dict)、命名元组 (namedtuple) 等容器组成的结构，容器里可以装叶子，也可以装其他节点。
#
# **为什么要发明这个概念？**
# 因为 JAX 的变换（`jit`, `grad`, `vmap`）都需要遍历输入和输出的所有张量。如果没有 PyTree 抽象，你只能处理单一数组。有了 PyTree，JAX 就可以对其进行“扁平化”和“还原”，从而让数学变换对任意复杂的 Python 结构生效。
#
# #### 2. 修炼核心：`jax.tree_util`
#
# 这是 JAX 中最常用的工具箱。作为修炼者，你需要掌握以下三个函数：
#
# ##### A. `tree_flatten` 与 `tree_unflatten` (解构与重构)
# 这是 JAX 内部视角的基石。

# %%
import jax
import jax.numpy as jnp
import numpy as np

# 一个典型的 PyTree：模型参数
params = {
    "layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.0])},
    "layer2": [jnp.array([3.0, 4.0]), jnp.array([5.0])],
}

# 1. 扁平化：把树拆成 "叶子列表" 和 "树的骨架 (treedef)"
leaves, treedef = jax.tree_util.tree_flatten(params)

print("Leaves (数据):", leaves)
# 输出: [Array([1., 2.],...), Array([0.],...), Array([3., 4.],...), Array([5.],...)]
print("TreeDef (结构):", treedef)
# 输出: PyTreeDef({'layer1': {'b': *, 'w': *}, 'layer2': [*, *]})

# 2. 还原：用骨架和叶子重新组装回去
restored_params = jax.tree_util.tree_unflatten(treedef, leaves)

print("检验还原是否成功:", str(params) == str(restored_params))
# %% [markdown]
# 我们可以看到，这里报了False，为什么呢？因为jax为了保证确定性，会自动按字典的 Key 进行字母排序。所以说从字符串的角度，还原前后参数并不是一致的。但是如果使用
# %%
print("检验还原是否成功:", params == restored_params)
# %% [markdown]
# 的话，python在比较两个字典是否相等时，只看集合属性，对顺序不敏感，所以会认为是相等的。
#
# **修炼点**：`treedef` 就像是模具，`leaves` 是填进去的铁水。JAX 的 `@jit` 编译时，只把 `leaves` 认为是动态数据，而 `treedef` 如果变了，会导致重新编译。
#
# ##### B. `tree_map` (结构映射)
# 这是你未来写代码最常用的函数。当你想要把所有参数乘以 0.1（学习率），或者对所有梯度做裁剪时，你不可能写无数个 `for` 循环。

# %%
# 对 PyTree 中的每一个叶子节点执行相同的函数
# 这里的 lambda x: x + 1 会作用在 params 里的每一个 array 上
new_params = jax.tree_util.tree_map(lambda x: x + 1, params)
print("原参数为：", params)
print("新参数为：", new_params)
# %% [markdown]
# **深度进阶**：
# `tree_map` 是支持**多参数**的！只要这些参数的结构（treedef）完全一致。
# `tree_map` 的工作原理：
# - 先看第一个参数 `params`，提取出结构模板（TreeDef）。
# - 然后，它假设后面传入的 `gradients` 也符合这个模板。
# - 它会同时遍历这两个树的叶子。
# %%
gradients = {
    "layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.0])},
    "layer2": [jnp.array([3.0, 4.0]), jnp.array([5.0])],
}
learning_rate = 0.01

# 传统的 SGD update step
updated_params = jax.tree_util.tree_map(
    lambda p, g: p - learning_rate * g, params, gradients
)
print("原参数为：", params)
print("更新后的参数为：", updated_params)

# %% [markdown]
# #### 3. 注册自定义类型 (Custom PyTree)
# 如果你写了一个自定义 Python 类，JAX 默认不认识它，会把它当成一个不可变的整体或者报错。你需要告诉 JAX 怎么拆解它。

# %%


@jax.tree_util.register_pytree_node_class
class MyLinear:
    def __init__(self, w, b, name):
        self.w = w
        self.b = b
        self.name = name  # name 是元数据，不是参与计算的张量

    # 告诉 JAX 哪些是孩子（参与微分/计算），哪些是辅助数（aux_data）
    def tree_flatten(self):
        children = (self.w, self.b)
        aux_data = self.name
        return children, aux_data

    # 告诉 JAX 怎么装回去
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)


# 现在 MyLinear 实例可以直接丢进 jit 或 grad 函数了


# %% [markdown]
# ---
#
# ### Part 2: JAX 的 PRNG 设计 —— 显式状态传递
#
# 这是从 PyTorch/NumPy 转 JAX 最不适应的地方。
#
# #### 1. 核心差异：Stateful vs Stateless
# *   **NumPy/PyTorch (Stateful)**:
#     有一个全局的随机数发生器（Global RNG）。每次你调用 `rand()`，它会在后台偷偷更新内部状态。
# %%
# NumPy

np.random.seed(0)
print(np.random.rand())  # 0.548...
print(np.random.rand())  # 0.715... (状态变了，但你看不见)
# %% [markdown]
# *   **JAX (Stateless)**:
#     没有全局状态。随机数由你传入的 `key` 根据哈希算法确定性生成的。**只要 Key 不变，结果永远一样**。
#
# #### 2. 修炼核心：`jax.random`
#
# ##### A. 为什么两次调用结果一样？
# %%

key = jax.random.PRNGKey(42)

print(jax.random.normal(key, shape=(1,)))  # 输出 A
print(jax.random.normal(key, shape=(1,)))  # 依然是输出 A！
# %% [markdown]
# **原理**：JAX 是函数式的，函数必须是 Pure 的。如果在函数内部依赖一个会变的全局变量，JAX 就无法对其进行可靠的变换（如并行化时，谁先拿随机数结果可能不同）。
#
# ##### B. `split` 机制 (裂变)
# 为了获得不同的随机数，你需要把 Key “裂变”成新的 Key。这就像细胞分裂。

# %%
key = jax.random.PRNGKey(42)

# 把一个 key 分裂成两个：
# current_key 用于这一步的操作
# next_key 留给传给下一层或者下一次迭代
current_key, next_key = jax.random.split(key, 2)

print(jax.random.normal(current_key))
print(jax.random.normal(next_key))  # 这两个就不一样了

# %% [markdown]
# ##### C. 多维 Split (大规模并行时的关键)
# 当你需要对一个 Batch 初始化参数时，不需要写循环 split。
# 目标：我们需要初始化 8 个独立的模型（Ensemble），或者处理一个 Batch 的数据。
# 核心问题：如何用一份代码，同时处理 N 份数据？

# %%
# 设置随机种子
master_key = jax.random.PRNGKey(42)

# 准备 8 个子 Key，代表我们要初始化的 8 个模型
keys = jax.random.split(master_key, num=8)
print(f"Keys shape: {keys.shape}")  # (8, 2)


# %% [markdown]
# 我们可以看到，`keys` 是一个形状为 (8, 2) 的数组，每一行都是一个独立的 PRNG Key，可以用来初始化不同的模型。而每个key本身是一个长度为2的数组，这是JAX PRNG Key的标准格式。
#
# **1. 为什么一个不够？（容量问题）**
# *   JAX 经常需要在并行的设备上生成海量的随机数。
# *   如果只用一个标准的 32 位整数（`uint32`），大约只有 40 亿（$2^{32}$）种可能的“种子”。
# *   在大规模分布式训练或者需要极高安全性的随机模拟中，40 亿其实很容易发生“碰撞”（即两个不该相同的随机流撞车了）。
#
# **2. 为什么要两个？（64位扩展）**
# *   JAX 实际上使用的是 **64 位** 的种子。
# *   $2^{64}$ 是一个天文数字（约 1844 亿亿），这保证了你在整个宇宙的生命周期里一直 split 下去，也很难遇到重复的 key。
# *   所以，这两个数合起来，其实代表了一个 **Huge Integer (64-bit)**。
#
# **3. 为什么不直接存成一个 64 位整数？（硬件兼容性）**
# *   JAX 的底层算法（Threefry-2x32）和很多 AI 硬件（特别是早期的 TPU 和部分 GPU 核心）并没有原生的 64 位整数处理单元，或者处理 32 位比 64 位快得多。
# *   于是，JAX 采取了一种工程上的折中：**把一个 64 位的大整数，“劈”成两个 32 位的整数存储**。
#     *   第一个数存“高 32 位”。
#     *   第二个数存“低 32 位”。
# *   这就好比如果你的提款机一次只能吐 100 块的纸币，你要取 200 块，它就给你吐两张，而不是印一张 200 面额的。

# %% [markdown]
# ### 0. 基础设定：单体函数
# 我们先写一个只负责初始化**一个**模型的函数。
# 这是最符合人类直觉的写法，完全不需要考虑 Batch 维度。


# %%
def init_single_layer(key, input_dim=10, output_dim=5):
    """
    输入一个 key，返回一组 {'w': ..., 'b': ...}
    """
    # 再次分裂，确保 w 和 b 用不同的随机源
    k1, k2 = jax.random.split(key)

    # 初始化参数
    w = jax.random.normal(k1, (input_dim, output_dim))
    b = jax.random.normal(k2, (output_dim,))

    return {"w": w, "b": b}


# 测试一下单体函数
single_param = init_single_layer(keys[0])
print("单个模型的 w shape:", single_param["w"].shape)
print("单个模型的 b shape:", single_param["b"].shape)

# %% [markdown]
# ### 1. The Hard Way (笨办法：Python 循环)
# 如果没有 JAX 的 vmap，我们通常会写一个 for 循环，得到一个字典列表 (List of Dicts)。
# 然后，为了能在 GPU 上高效计算，我们通常还需要手动把它们堆叠起来。

# %%
# 1. 循环生成
list_of_params = []
for k in keys:
    list_of_params.append(init_single_layer(k))

print(f"笨办法得到的数据类型: type(list) -> {type(list_of_params[0])}")
print("目前结构: [{'w':.., 'b':..}, ..., {'w':.., 'b':..}] (Array of Structures)")

# %% [markdown]
# 2. 手动堆叠 (为了变成 GPU 喜欢的 Tensor 格式)
# 这一步通常很痛苦，需要再次遍历字典结构
# 这里我们借用 tree_map 模拟一下手动堆叠的过程[[day01-jax与纯函数#B.  (结构映射)|tree map]]

# %%
stacked_params_manual = jax.tree_util.tree_map(
    lambda *args: jnp.stack(args), *list_of_params
)

print("手动堆叠后的 w shape:", stacked_params_manual["w"].shape)
# 结果应该是 (8, 10, 5)

# %% [markdown]
# ### 2. The JAX Way (自动挡：Vmap)
# 现在我们使用 `jax.vmap`。
# 注意：我们没有修改 `init_single_layer` 的任何一行代码！
# vmap 就像一个转换器，把“处理单个样本的函数”变成了“处理一批样本的函数”。

# %%
# 定义转换后的函数
# init_single_layer: (key) -> dict
# batched_init:      (keys[Batch]) -> dict[Batch]
batched_init = jax.vmap(init_single_layer)

# 执行
batched_params_vmap = batched_init(keys)

print("Vmap 自动生成的 w shape:", batched_params_vmap["w"].shape)
print("Vmap 自动生成的 b shape:", batched_params_vmap["b"].shape)

# %% [markdown]
# ### 3. 验证与结构分析 (PyTree 视角)
# 让我们看看这两者是否一致，并深入理解数据结构的变化。

# %%
# 验证数值完全一致
# 注意：前提是随机数 Key 的顺序和使用方式一致
assert jnp.allclose(stacked_params_manual["w"], batched_params_vmap["w"])
assert jnp.allclose(stacked_params_manual["b"], batched_params_vmap["b"])
print("验证通过：笨办法和 Vmap 得到的结果数值完全一致。")

# 深入结构分析
print("\n--- 结构对比 (The Aha! Moment) ---")
print("1. 人类的直觉 (Array of Structures):")
print("   List [ Model1{'w','b'}, Model2{'w','b'} ... ]")
print("   这种存储在内存里是不连续的，GPU 读起来很慢。")

print("\n2. GPU 的直觉 (Structure of Arrays):")


# 打印一下 Vmap 结果的树结构
def print_shape(x):
    return x.shape


structure = jax.tree_util.tree_map(print_shape, batched_params_vmap)
print(f"   Dict {{ 'w': BatchArray{structure['w']}, 'b': BatchArray{structure['b']} }}")
print("   Vmap 自动帮我们把 Batch 维度放在了最前面 (Leading Dimension)。")
print("   这意味着所有模型的 'w' 都在一块连续的内存里，可以一个指令并行计算所有模型！")
