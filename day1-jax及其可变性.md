### Part 1: PyTree —— JAX 的通用数据容器

在 PyTorch 中，我们习惯于 `model(x)`，但在 JAX 中，我们经常看到 `apply_fn(params, x)`。这里的 `params` 通常是一个复杂的嵌套字典，这就是 PyTree 的一种。

#### 1. 什么是 PyTree？
简单来说，**PyTree 是 Python 容器结构的抽象**。它是一个递归定义的概念：
*   **叶子节点 (Leaves)**：各种数组（JAX arrays, NumPy arrays）、标量、或者 `None`。
*   **节点 (Nodes)**：由列表 (list)、元组 (tuple)、字典 (dict)、命名元组 (namedtuple) 等容器组成的结构，容器里可以装叶子，也可以装其他节点。

**为什么要发明这个概念？**
因为 JAX 的变换（`jit`, `grad`, `vmap`）都需要遍历输入和输出的所有张量。如果没有 PyTree 抽象，你只能处理单一数组。有了 PyTree，JAX 就可以对其进行“扁平化”和“还原”，从而让数学变换对任意复杂的 Python 结构生效。

#### 2. 修炼核心：`jax.tree_util`

这是 JAX 中最常用的工具箱。作为修炼者，你需要掌握以下三个函数：

##### A. `tree_flatten` 與 `tree_unflatten` (解构与重构)
这是 JAX 内部视角的基石。

```python
import jax
import jax.numpy as jnp

# 一个典型的 PyTree：模型参数
params = {
    'layer1': {'w': jnp.array([1., 2.]), 'b': jnp.array([0.])},
    'layer2': [jnp.array([3., 4.]), jnp.array([5.])]
}

# 1. 扁平化：把树拆成 "叶子列表" 和 "树的骨架 (treedef)"
leaves, treedef = jax.tree_util.tree_flatten(params)

print("Leaves (数据):", leaves) 
# 输出: [Array([1., 2.],...), Array([0.],...), Array([3., 4.],...), Array([5.],...)]
print("TreeDef (结构):", treedef)
# 输出: PyTreeDef({'layer1': {'b': *, 'w': *}, 'layer2': [*, *]})

# 2. 还原：用骨架和叶子重新组装回去
restored_params = jax.tree_util.tree_unflatten(treedef, leaves)
assert str(params) == str(restored_params)
```
**修炼点**：`treedef` 就像是模具，`leaves` 是填进去的铁水。JAX 的 `@jit` 编译时，只把 `leaves` 认为是动态数据，而 `treedef` 如果变了，会导致重新编译。

##### B. `tree_map` (结构映射)
这是你未来写代码最常用的函数。当你想要把所有参数乘以 0.1（学习率），或者对所有梯度做裁剪时，你不可能写无数个 `for` 循环。

```python
# 对 PyTree 中的每一个叶子节点执行相同的函数
# 这里的 lambda x: x + 1 会作用在 params 里的每一个 array 上
new_params = jax.tree_util.tree_map(lambda x: x + 1, params)
```
**深度进阶**：
`tree_map` 是支持**多参数**的！只要这些参数的结构（treedef）完全一致。
```python
gradients = ... # 一个结构和 params 完全一样的梯度树
learning_rate = 0.01

# 传统的 SGD update step
updated_params = jax.tree_util.tree_map(
    lambda p, g: p - learning_rate * g, 
    params, 
    gradients
)
```

#### 3. 注册自定义类型 (Custom PyTree)
如果你写了一个自定义 Python 类，JAX 默认不认识它，会把它当成一个不可变的整体或者报错。你需要告诉 JAX 怎么拆解它。

```python
from functools import partial

@jax.tree_util.register_pytree_node_class
class MyLinear:
    def __init__(self, w, b, name):
        self.w = w
        self.b = b
        self.name = name # name 是元数据，不是参与计算的张量
    
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
```

---

### Part 2: JAX 的 PRNG 设计 —— 显式状态传递

这是从 PyTorch/NumPy 转 JAX 最痛苦的地方。

#### 1. 核心差异：Stateful vs Stateless
*   **NumPy/PyTorch (Stateful)**: 
    有一个全局的随机数发生器（Global RNG）。每次你调用 `rand()`，它会在后台偷偷更新内部状态。
    ```python
    # NumPy
    np.random.seed(0)
    print(np.random.rand()) # 0.548...
    print(np.random.rand()) # 0.715... (状态变了，但你看不见)
    ```
*   **JAX (Stateless)**: 
    没有全局状态。随机数由你传入的 `key` 根据哈希算法确定性生成的。**只要 Key 不变，结果永远一样**。

#### 2. 修炼核心：`jax.random`

##### A. 为什么两次调用结果一样？
```python
import jax

key = jax.random.PRNGKey(42)

print(jax.random.normal(key, shape=(1,))) # 输出 A
print(jax.random.normal(key, shape=(1,))) # 依然是输出 A！
```
**原理**：JAX 是函数式的，函数必须是 Pure 的。如果在函数内部依赖一个会变的全局变量，JAX 就无法对其进行可靠的变换（如并行化时，谁先拿随机数结果可能不同）。

##### B. `split` 机制 (裂变)
为了获得不同的随机数，你需要把 Key “裂变”成新的 Key。这就像细胞分裂。

```python
key = jax.random.PRNGKey(42)

# 把一个 key 分裂成两个：
# current_key 用于这一步的操作
# next_key 留给传给下一层或者下一次迭代
current_key, next_key = jax.random.split(key, 2)

r1 = jax.random.normal(current_key)
r2 = jax.random.normal(next_key) # 这两个就不一样了
```

##### C. 多维 Split (大规模并行时的关键)
当你需要对一个 Batch 初始化参数时，不需要写循环 split。
```python
keys = jax.random.split(key, num=8) # 产生 8 个 subkeys
```
配合 `vmap` 使用是 JAX 的经典操作：
```python
# 假设 init_func(key) 负责初始化一个样本的参数
batched_init = jax.vmap(init_func)
params = batched_init(keys) # 并行初始化 8 套参数
```
