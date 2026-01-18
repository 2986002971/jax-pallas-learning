# Day 5: 状态管理的艺术 (The Art of State Management)

**核心叙事**：在 Day 4 中，我们掌握了自动微分这把手术刀。现在我们要造车了。
但在 JAX 的世界里，造车面临一个巨大的哲学冲突：神经网络本质是**有状态的**（它有权重、有记忆），而 JAX 的变换（JIT/Grad/Vmap）要求函数是**无状态的**（Pure Function）。

今天，我们将从“裸 JAX”开始，亲手体验搬运参数的痛苦，从而引出 Flax 等框架存在的真正意义：**在面向对象的定义和函数式的执行之间，建立一座桥梁。**

## Part 1: 哲学之辩 —— 为什么 Google 抛弃了面向对象？

在开始写代码之前，我们需要回答一个几乎所有从 PyTorch 转到 JAX 的开发者都会问的问题：
> *"为什么 Flax 这么反人类？为什么每次 forward 都要显式地把 params 传进去？为什么不能像 PyTorch 那样直接 `self.layer(x)`？"*

这并不是 JAX 的设计缺陷，这是一个**为了 Scale（规模化）而做出的妥协**。

对于习惯了 PyTorch 的开发者来说，Flax 的写法显得有些啰嗦。但 Google 的工程师团队（包括 PaLM, Gemini 的研发团队）坚定地选择这种“数据与逻辑分离”的架构，核心原因在于：**当系统规模扩大到千卡、万卡分布式训练，以及面临严苛的 MLOps 管理时，纯函数架构拥有压倒性的优势。**

### 1. 拒绝“幽灵状态” (No Ghost State)

在大规模工程中，最可怕的 Bug 是“不可复现的 Bug”，这往往是**隐式状态修改**导致的。

*   **OOP 的隐患**：在 PyTorch 中，执行 `model(x)` 可能会在内部悄悄改变状态。
    *   比如 `BatchNorm` 默默更新了 `running_mean`。
    *   比如 `Dropout` 依赖全局随机种子。
    *   在多卡并行时，如果不同的卡修改了同一个对象的内部状态，或者忘记切换 `model.eval()`，会导致灾难性的**竞态条件 (Race Condition)**，这种 Bug 极难排查。
*   **JAX 的安全感**：纯函数意味着**零副作用**。
    *   `predict(params, x)` 绝对不会修改 `params` 里的任何东西。
    *   新的参数只能通过 `new_params = optimizer(old_params, grads)` 显式生成。
    *   这种**不可变性 (Immutability)** 让大系统的调试变得像数学推导一样可靠：只要输入确定，输出永远确定。

### 2. 参数切片与分布式 (Sharding Is Data Manipulation)

在大模型时代，一个模型的参数（比如 70B）根本放不进单张显卡。

*   **OOP 的困境**：参数藏在对象内部（`self.w`）。你想做模型并行（Tensor Parallelism），通常需要给模型代码打补丁（Hooks），代码侵入性很强。
*   **FP 的优势**：在 JAX/Flax 中，参数只是一个**纯粹的数据字典**（PyTree）。
    *   模型代码根本不需要知道分布式的存在。
    *   工程师可以直接对这个字典进行操作（切分、重组），使用 `jax.sharding` 定义哪一层权重放在哪些 TPU/GPU 上。
    *   **数据与逻辑解耦**：你可以在完全没有实例化模型类的情况下，直接加载和处理 Checkpoint 数据。

### 3. 完美契合编译器 (Compiler Friendliness)

JAX 的底层是 XLA（加速线性代数编译器）。XLA 本质上是一个**函数式图编译器**。

它最喜欢处理的就是这种结构：`Output = Function(Input1, Input2)`。
当代码结构是纯函数时，XLA 可以放心地进行**极端的大规模算子融合**，因为它知道这个函数内部没有“黑魔法”（比如读取全局变量）。Flax 的写法就是顺着 XLA 的“毛”来摸，能把 TPU 集群的算力榨干到极致。

---

### 总结：作坊 vs 工厂

*   **PyTorch (OOP)**：像是在开**单体作坊**。工具都在手边（`self.something`），写起来快，直觉顺畅，极其适合研究者、中小型模型和快速验证。
*   **JAX/Flax (FP)**：像是在建**现代化流水线工厂**。工人（函数）每次拿材料（`params`）都要填出库单（显式传参），看起来繁琐，但这保证了**极高的容错率、可追踪性和无限的横向扩展能力**。

---

## Part 2: 荒野求生 —— 徒手使用 Raw JAX

现在，让我们脱掉 Flax/Haiku/Optax 这些防护服，赤手空拳地用最原始的 `jax.numpy` 来构建一个神经网络。
只有体验过这种“显式管理一切”的痛苦，你才能真正学会 JAX。

### 2.1 参数初始化：字典地狱

在 Raw JAX 中，没有 `nn.Linear` 帮你自动初始化。你必须自己管理随机数钥匙（PRNGKey）和参数形状。

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import random
```

```python
# 定义一个简单的 MLP 结构：Input(2) -> Hidden(10) -> Output(1)
def init_mlp_params(key):
    # JAX 的随机数必须显式分裂 (Split)
    k1, k2 = random.split(key)

    # 第一层权重: 输入 2 -> 输出 10
    w1 = random.normal(k1, (2, 10)) * jnp.sqrt(2 / 2)  # He初始化
    b1 = jnp.zeros(10)

    # 第二层权重: 输入 10 -> 输出 1
    w2 = random.normal(k2, (10, 1)) * jnp.sqrt(2 / 10)
    b2 = jnp.zeros(1)

    # 核心：参数只是一个嵌套的字典 (PyTree)
    return {"layer1": {"w": w1, "b": b1}, "layer2": {"w": w2, "b": b2}}


key = random.PRNGKey(42)
params = init_mlp_params(key)

# 打印看看，真的只是个字典
print(jax.tree_util.tree_structure(params))
# 输出: PyTreeDef({'layer1': {'b': *, 'w': *}, 'layer2': {'b': *, 'w': *}})
```

**痛点 1**：必须手动计算每一层的 `fan_in` 和 `fan_out`。如果我把中间层从 10 改成 20，我得改两处代码（第一层的输出维，和第二层的输入维）。如果在 PyTorch 里，这只是 `nn.Linear(20, 1)` 的事。

### 2.2 前向传播：手动解包

在定义前向传播时，我们不能使用 `self.w`。我们必须把 `params` 字典传进去，然后一层层剥开。




```python
def predict(params, x):
    # 显式解包：从字典里拿到需要的权重
    w1 = params["layer1"]["w"]
    b1 = params["layer1"]["b"]
    w2 = params["layer2"]["w"]
    b2 = params["layer2"]["b"]

    # 计算逻辑
    hidden = jnp.dot(x, w1) + b1
    hidden = jax.nn.relu(hidden)  # Activation

    output = jnp.dot(hidden, w2) + b2
    return output


# 测试一下
x_dummy = jnp.ones((5, 2))  # Batch size 5
y_pred = predict(params, x_dummy)
print(f"Prediction shape: {y_pred.shape}")
```


**痛点 2**：如果网络有 100 层，这个 `predict` 函数会写到手断。而且，如果我想复用这个 Layer，我得自己写函数来管理字典的 key 路径。

### 2.3 训练循环：手写优化器

没有 `optimizer.step()`。我们需要利用 `jax.tree_util` 来批量更新字典里的每一个叶子节点。[[day01-jax与纯函数]]  [[day04-自动微分]]




```python
# 1. 定义 Loss
def loss_fn(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)


# 2. 编译 Update Step
@jax.jit
def update_step(params, x, y, learning_rate=0.01):
    # 自动微分
    loss_val, grads = jax.value_and_grad(loss_fn)(params, x, y)

    # 手写 SGD：params = params - lr * grads
    # tree_map 是 JAX 中处理参数字典的神器
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grads
    )

    return new_params, loss_val


# 3. 模拟训练
x_data = random.normal(random.PRNGKey(0), (100, 2))
y_data = random.normal(random.PRNGKey(1), (100, 1))

for i in range(5):
    params, loss_val = update_step(params, x_data, y_data)
    print(f"Step {i}, Loss: {loss_val:.4f}")
```


**痛点 3**：这只是 SGD。如果要实现 Adam 呢？你需要维护动量 `m` 和 `v`。这意味着除了 `params`，你还得再搞一个结构完全一样的 `opt_state` 字典来存动量，然后在 `update_step` 里同时输入、更新并返回这个状态。

### 2.4 深度思考：如果有 Batch Norm 怎么办？

这是 Raw JAX 最让人头秃的地方。
`Batch Norm` 在训练时需要更新 `running_mean` 和 `running_var`。但在 JAX 里，你不能直接改 `params['bn']['mean']`。

你必须把 `predict` 函数改成这样：

```pseudocode
# 伪代码
def predict(params, batch_stats, x, is_training):
    # ... 计算 ...
    if is_training:
        new_running_mean = ...
        return y, new_batch_stats
    else:
        return y, batch_stats
```

这意味着你的 Loss 函数签名变了，你的 `jax.grad` 必须处理辅助返回值（`has_aux=True`），你的训练循环必须显式地接收并传递 `new_batch_stats` 给下一步。
**状态流转的复杂性，会随着网络复杂度的增加呈指数级上升。**

---

## 阶段性总结

通过这段“荒野求生”，我们发现：
1.  **JAX 的底层非常干净**：就是矩阵计算 + 字典操作。
2.  **手动管理太累了**：初始化形状推导、随机 Key 的传递、复杂状态（BN/Dropout）的流转、优化器的状态管理，每一项都是繁重的体力活。

这就是为什么我们需要 **Flax**。

Flax 不是要把 JAX 变回 PyTorch，它是**为了让你能用面向对象的方式来定义“结构”，但依然以函数式的方式来“执行”计算。**

接下来，我们将揭开 Flax 的魔法 —— 它是如何通过 Scope 机制帮我们自动管理这些讨厌的字典 key 和状态流转的。

我们将进入 Day 5 的核心部分：**Flax 的设计哲学**。

不同于 PyTorch 的“所见即所得”，Flax 采用了一种**“蓝图与实体分离”**的模式。这部分内容是理解 Flax 乃至整个 JAX 生态中最反直觉、但也最精彩的“Aha Moment”。

---

## Part 3: 秩序的建立 —— Flax 的二象性

**核心叙事**：Flax 并没有改变 JAX“纯函数”的本质。它的作用就像是一个**高明的会计**。
它允许你用面向对象（Class）的方式画图纸，但在执行时，它会自动把对象拆解成 JAX 能够理解的纯数据（Params）和纯逻辑（Function）。

### 3.1 蓝图 (Module) 与 实体 (Variables)

在 PyTorch 中，当你实例化一个 Layer 时，显存就被占用了：

```pytorch
# PyTorch: 实例化即创建
layer = nn.Linear(10, 20)  # 权重矩阵立即被创建并在内存中分配
y = layer(x)
```

在 Flax 中，实例化一个 Layer 仅仅是画了一张**图纸**（Configuration）：

```flax
# Flax: 只是个轻量级的 dataclass
layer = nn.Dense(features=20) # 没有权重被创建！连输入维度都没指定！
```

直到你提供数据（Input）和钥匙（RNG Key）并调用 `init` 方法时，Flax 才会真正去显存里开辟空间：

```flax
params = layer.init(key, x)   # 这才是"实体化"的过程
y = layer.apply(params, x)    # 使用实体数据去驱动蓝图逻辑
```

这种**惰性初始化 (Lazy Initialization)** 解决了 Part 1 中的第一个痛点：我们再也不需要手动计算 `fan_in`（输入维度）了！Flax 会等到看见输入数据 $x$ 的那一刻，根据 $x.shape$ 自动推导权重的形状。

### 3.2 实战：重构 MLP

让我们用 Flax 重写之前的 MLP。请注意 `@nn.compact` 装饰器，它是 Flax 的魔法核心。



```python
class MLP(nn.Module):
    # 超参数作为类的字段 (Fields)
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        # 魔法时刻：我们在 forward pass 里直接定义层！
        # 只有在第一次调用 init 时，这些层才会被初始化。
        # 这里的 name='dense1' 对应 params 字典里的 key
        x = nn.Dense(features=self.hidden_dim, name="dense1")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_dim, name="dense2")(x)
        return x


# 1. 实例化蓝图 (Config)
model = MLP(hidden_dim=10, out_dim=1)

# 2. 实体化 (Init) - 也就是 Trace 一遍
# 这里的 x_dummy 用于告诉 Flax 输入维度是 2
key1, key2 = random.split(random.PRNGKey(0))
x_dummy = jnp.ones((1, 2))

variables = model.init(key1, x_dummy)  # 返回一个 PyTree

print("=== Flax 生成的参数结构 ===")
print(variables)
# 输出结构清晰的字典：
# {
#   'params': {
#     'dense1': {'kernel': ..., 'bias': ...},
#     'dense2': {'kernel': ..., 'bias': ...}
#   }
# }
```

**解析**：
*   **无需指定输入维度**：`nn.Dense` 只需知道输出维度。输入维度由 `x_dummy` 隐式决定。
*   **结构自动分层**：Flax 自动根据代码结构（`dense1`, `dense2`）生成了嵌套字典。

### 3.3 变量集合 (Variable Collections) —— 解决 BN 痛点

在 Part 1 中，最让我们头疼的是 Batch Norm 的状态管理（Running Mean/Var 不是参数，不需要梯度，但需要更新）。

Flax 发明了 **Collections** 来解决这个问题。它把所有的状态分门别类：
1.  **params**: 需要梯度下降更新的（Weights, Biases）。
2.  **batch_stats**: 前向传播时顺手更新的（Running Mean/Var）。
3.  **cache**: 自回归生成时的 KV Cache。
4.  **perturbations**: 对抗训练时的扰动。

让我们看一个包含 BN 的例子，这会让你理解 `model.apply` 的完全体形态。



```python
class BN_MLP(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool):
        # use_running_average: 训练时为 False (用当前 Batch 统计量)，预测时为 True
        x = nn.BatchNorm(use_running_average=not training, name="bn1")(x)
        x = nn.Dense(1, name="head")(x)
        return x


model_bn = BN_MLP()

# 初始化时，Flax 会同时生成 params 和 batch_stats
variables = model_bn.init(key1, x_dummy, training=False)

print("\n=== 含状态的变量结构 ===")
print(variables.keys())
# output: dict_keys(['params', 'batch_stats'])
```

**关键来了：状态变异 (Mutation) 的函数式表达**

在训练循环中，我们需要告诉 `apply` 函数：“不仅要算输出 $y$，还要把更新后的 `batch_stats` 吐出来”。

```python
# mutable=['batch_stats']: 告诉 Flax，这个集合允许被修改
# 这里的 variables 包含了 params 和 batch_stats
y, new_state = model_bn.apply(
    variables, x_dummy, training=True, mutable=["batch_stats"]
)

# update_state 里只包含 batch_stats，不包含 params
print("返回的新状态 keys:", new_state.keys())
```

**对比 Part 1**：
*   我们不再需要手动写 `if is_training: return y, new_stats`。
*   Flax 统一了接口：只要传入 `mutable=['...']`，它就会把副作用变成返回值。

---

## Part 4: 深度解剖 —— Scope 机制 (The Invisible Cursor)

*这一节是为了满足那些想知道“到底怎么做到的”硬核读者。*

<details>
你可能会好奇：在 `nn.Dense` 的代码里，它既要在初始化时**创建**参数，又要在前向传播时**读取**参数，而且这一切都发生在同一个 `__call__` 方法里。它是怎么知道当前该干什么的？它又是怎么知道去哪里找参数的？

答案就在源码中无处不在的 **Scope** 对象里。

### 4.1 核心隐喻：公式 $f$ 与系数 $\theta$

在 PyTorch 中，公式和系数是粘在一起的。当你创建一个 Layer 对象，参数张量就长在这个对象身上。

在 Flax 中，它们是彻底分离的：
1.  **Module (公式)**：当你实例化 `layer = nn.Dense(10)` 时，查看源码的 `__init_subclass__` 和 `dataclass` 定义，你会发现**没有任何 JAX 数组被创建**。你只是定义了一个数学算子的**结构描述**（比如：这里需要一个乘法，那里需要一个 Bias 加法）。
2.  **Variables (系数)**：真正的数值存储在一个外部的、巨大的嵌套字典（PyTree）里。

那么，是谁把这两个世界联系起来的？是 **Scope**。
Scope 就像是一个**游标 (Cursor)** 或**读写磁头**，它拿着一张巨大的地图（Variables 字典），在 Module 构成的数学公式树中穿梭，告诉每一个算子：“你的 $w$ 在字典的第 3 行，你的 $b$ 在第 4 行”。

### 4.2 源码解密 A：Module 只是个“空壳”

在源码的 `Module` 类定义中，最精彩的是 `__post_init__` 方法。

```flax
# Flax 源码简化逻辑
def __post_init__(self):
    # 1. 寻找父亲
    if self.parent is _unspecified_parent:
        self.parent = _context.module_stack[-1] # 自动认亲

    # 2. 注册 Scope (这是关键！)
    if isinstance(self.parent, Module):
        # 向父亲索要一个新的 Scope，并在路径上加上自己的名字
        object.__setattr__(
            self, 'scope', self.parent.scope.push(self.name)
        )
```

**解析**：
当你写 `x = nn.Dense(10, name='fc1')(x)` 时：
1.  Flax 检查当前的上下文栈（Stack），发现你正在 `MyMLP` 内部。
2.  它自动把 `MyMLP` 设为 `Dense` 的父节点。
3.  它调用 `parent.scope.push('fc1')`。这相当于在文件路径中由 `/MyMLP/` 进入了 `/MyMLP/fc1/`。
4.  这个新的 Scope 被绑定到了 `Dense` 实例上。现在，这个 `Dense` 实例拥有了访问全局字典中 `'fc1'` 子树的权限。

### 4.3 源码解密 B：`apply` 的移花接木

为什么必须通过 `model.apply(params, x)` 来调用模型？看源码中的 `apply` 函数实现：

```flax
def apply(fn, module, variables, ...):
    # 1. 基于 variables 创建根 Scope
    root_scope = Scope(variables, ...)

    # 2. 核心魔法：克隆 Module 并注入 Scope
    # 原来的 module 是没有 scope 的（Unbound），现在我们克隆一个有 scope 的（Bound）
    bound_module = module.clone(parent=root_scope)

    # 3. 执行前向传播
    return fn(bound_module, ...)
```

**解析**：
这证实了 Flax 的 **Functional** 本质。
你手里的那个 `model` 对象其实是一个**模具**。每次调用 `apply`，Flax 都会根据这个模具，配合传入的 `variables`，现场组装出一个临时的、带有数据的**实体对象** (`bound_module`) 来执行计算。计算一结束，这个实体对象就被丢弃了。

### 4.4 源码解密 C：`param` 的双重人格

我们在写 Layer 时常用的 `self.param('kernel', ...)`，到底在干什么？
看源码：

```flax
def param(self, name, init_fn, ...):
    # 委托给 Scope 去办
    return self.scope.param(name, init_fn, ...)
```

`Scope.param` 的内部逻辑极其精妙，它解决了“初始化”和“使用”的二象性：

*   **Scenario 1: 初始化阶段 (Init)**
    Scope 拿着 `init_fn`（初始化函数），发现全局字典里对应的位置是空的。于是它运行 `init_fn(key)` 生成随机数组，将其**填入**字典，并返回这个新数组。

*   **Scenario 2: 运行阶段 (Apply)**
    Scope 发现全局字典里已经有东西了。它直接忽略 `init_fn`，把字典里的那个数组**取出来**返回给你。

这就解释了为什么我们的代码里不用写 `if initialized: return w else: create w`。Scope 把这个 dirty work 完美地封装了。

### 4.5 总结

读完源码，我们终于看清了 Flax 的真面目：

*   **Module** 是公式及其层级结构。
*   **Context Stack (`_context`)** 负责在定义网络时自动维护父子关系，让我们不用手动传递 parent。
*   **Scope** 是一个状态机。它在 `init` 时是**录入员**（把系数填入字典），在 `apply` 时是**图书管理员**（按索引把系数取出来给公式使用）。

这种设计虽然让 API 看起来比 PyTorch 繁琐（因为显式分离了 State），但它换来的是**绝对的数学纯洁性**。这正是 JAX 能够随意进行 `vmap`（批处理化）、`pmap`（并行化）和 `grad`（求导）的根本原因——因为公式 $f$ 本身不再持有任何状态，它只是一个纯粹的计算描述符。
</details>

---

## Part 5: 工业级标准 —— TrainState 与 Optax

至此，我们有了模型定义，也有了参数管理。最后一步，是把优化器（Optimizer）也集成进来，打包成一个标准的训练状态容器。

这就要用到 `flax.training.train_state` 和 DeepMind 的优化库 `optax`。

### 5.1 为什么要 TrainState？

在 Part 1 中，我们的 `update_step` 需要分别传入 `params` 和 `opt_state`（如果用 Adam）。如果以后还要加 `batch_stats`，函数签名会越来越长。
`TrainState` 就是一个 dataclass，它把这些东西打包带走，方便 JIT 编译。

### 5.2 终极代码：50 行搞定 MLP 训练

这是你以后写 JAX 项目的标准模板。

```python
# 1. 定义模型
model = MLP(hidden_dim=10, out_dim=1)


# 2. 创建 TrainState (工厂模式)
def create_train_state(rng, learning_rate):
    # 初始化参数
    params = model.init(rng, jnp.ones((1, 2)))["params"]

    # 定义优化器 (Optax 是纯函数式的)
    tx = optax.adam(learning_rate)

    # 打包！
    # TrainState 会自动帮我们调用 tx.init(params) 生成 opt_state
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# 3. 定义 Update Step (这就是被 JIT 的那个函数)
@jax.jit
def train_step(state, batch_x, batch_y):
    # state 包含了 params, opt_state 和 apply_fn

    def loss_fn(params):
        # 使用 state.apply_fn 代替直接调用 model.apply
        pred = state.apply_fn({"params": params}, batch_x)
        return jnp.mean((pred - batch_y) ** 2)

    # 自动微分
    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # 更新状态
    # state.apply_gradients 内部自动做了:
    # 1. updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
    # 2. new_params = optax.apply_updates(state.params, updates)
    # 3. 返回一个新的 state 对象
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss


# --- 运行 ---
rng = random.PRNGKey(0)
state = create_train_state(rng, learning_rate=0.01)

# 模拟数据
x = random.normal(rng, (32, 2))
y = random.normal(rng, (32, 1))

# 训练五步
for step in range(5):
    state, loss = train_step(state, x, y)
    print(f"Step {step}, Loss: {loss:.4f}")
```

### 总结

Day 5 我们完成了从“手耕火种”到“机械化农业”的转变。

1.  **Raw JAX**: 让我们明白了神经网络本质就是 `y = f(params, x)`，但手动管理 `params` 是地狱。
2.  **Flax Module**: 帮我们将逻辑分层（OOP），并通过 `init/apply` 机制将其映射回 JAX 的纯数据流（FP）。
3.  **TrainState & Optax**: 提供了标准化的状态容器和优化器接口，消除了样板代码。

至此，你已经掌握了 JAX 的**计算核心 (Day 1-3)**、**微分原理 (Day 4)** 和 **工程架构 (Day 5)**。
