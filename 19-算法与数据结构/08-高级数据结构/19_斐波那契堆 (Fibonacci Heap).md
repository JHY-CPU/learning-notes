# 斐波那契堆 (Fibonacci Heap)

## 1. 概述

斐波那契堆（Fibonacci Heap）是由 Michael Fredman 和 Robert Tarjan 于 1984 年发明的一种可并堆数据结构。它以**势能分析**为基础，实现了多个操作的优异均摊复杂度。

核心优势：插入和合并操作的均摊复杂度为 O(1)，这使得它在某些算法（如 Dijkstra）中表现出色。

## 2. 结构设计

### 2.1 整体结构

斐波那契堆由多棵**最小堆有序树**组成，用**双向循环链表**连接。

```
斐波那契堆结构：

  min
   |
   v
  [3] <-> [7] <-> [1] <-> [18]    (根链表)
   |        |       |
  [5]     [9]     [2]
   |              / \
  [8]           [4] [6]
```

### 2.2 节点结构

每个节点包含：
- key：键值
- degree：子节点个数
- mark：标记位（是否失去过子节点）
- parent, child, left, right：指针

### 2.3 Python 实现

```python
class FibNode:
    """斐波那契堆节点"""
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.mark = False
        self.parent = None
        self.child = None
        self.left = self  # 双向循环链表
        self.right = self
```

### 2.4 C++ 实现

```cpp
struct FibNode {
    int key;
    int degree;
    bool mark;
    FibNode* parent;
    FibNode* child;
    FibNode* left;
    FibNode* right;

    FibNode(int k) : key(k), degree(0), mark(false),
                     parent(nullptr), child(nullptr) {
        left = right = this;
    }
};
```

## 3. 核心操作

### 3.1 插入（O(1)）

将新节点加入根链表，如果更小则更新 min 指针。

```python
def insert(self, key):
    """插入新节点，均摊 O(1)"""
    node = FibNode(key)
    if self.min_node is None:
        self.min_node = node
    else:
        # 加入根链表
        self._add_to_root_list(node)
        if node.key < self.min_node.key:
            self.min_node = node
    self.n += 1
    return node

def _add_to_root_list(self, node):
    """将节点加入根链表"""
    node.left = self.min_node
    node.right = self.min_node.right
    self.min_node.right.left = node
    self.min_node.right = node
```

### 3.2 合并（O(1)）

将两个堆的根链表拼接。

```python
def merge(self, other):
    """合并两个斐波那契堆，O(1)"""
    if other.min_node is None:
        return
    if self.min_node is None:
        self.min_node = other.min_node
    else:
        # 拼接两个根链表
        self_last = self.min_node.left
        other_last = other.min_node.left

        self_last.right = other.min_node
        other.min_node.left = self_last
        other_last.right = self.min_node
        self.min_node.left = other_last

        if other.min_node.key < self.min_node.key:
            self.min_node = other.min_node

    self.n += other.n
```

### 3.3 获取最小值（O(1)）

```python
def find_min(self):
    """获取最小值，O(1)"""
    return self.min_node
```

### 3.4 删除最小值（O(log n) 均摊）

1. 将 min 节点的子节点全部加入根链表
2. 删除 min 节点
3. **合并相同度数的根节点**（consolidate）

```python
def extract_min(self):
    """删除并返回最小值，均摊 O(log n)"""
    z = self.min_node
    if z is not None:
        # 将 z 的子节点加入根链表
        if z.child is not None:
            children = self._get_all_nodes(z.child)
            for child in children:
                child.parent = None
                self._add_to_root_list(child)

        # 从根链表移除 z
        self._remove_from_root_list(z)

        if z == z.right:
            self.min_node = None
        else:
            self.min_node = z.right
            self._consolidate()

        self.n -= 1
    return z

def _consolidate(self):
    """合并相同度数的根节点"""
    max_degree = int(self.n ** 0.5) + 1
    degree_table = [None] * (max_degree + 1)

    nodes = self._get_all_nodes(self.min_node)

    for w in nodes:
        x = w
        d = x.degree
        while d < len(degree_table) and degree_table[d] is not None:
            y = degree_table[d]
            if x.key > y.key:
                x, y = y, x
            self._heap_link(y, x)
            degree_table[d] = None
            d += 1
        if d < len(degree_table):
            degree_table[d] = x

    # 重建根链表和 min 指针
    self.min_node = None
    for node in degree_table:
        if node is not None:
            if self.min_node is None:
                self.min_node = node
                node.left = node.right = node
            else:
                self._add_to_root_list(node)
                if node.key < self.min_node.key:
                    self.min_node = node

def _heap_link(self, y, x):
    """将 y 作为 x 的子节点"""
    self._remove_from_root_list(y)
    y.parent = x
    if x.child is None:
        x.child = y
        y.left = y.right = y
    else:
        y.left = x.child
        y.right = x.child.right
        x.child.right.left = y
        x.child.right = y
    x.degree += 1
    y.mark = False
```

### 3.5 减小键值（O(1) 均摊）

```python
def decrease_key(self, node, new_key):
    """减小键值，均摊 O(1)"""
    if new_key > node.key:
        raise ValueError("新键值不能大于原键值")

    node.key = new_key
    parent = node.parent

    if parent is not None and node.key < parent.key:
        self._cut(node, parent)
        self._cascading_cut(parent)

    if node.key < self.min_node.key:
        self.min_node = node

def _cut(self, x, y):
    """将 x 从 y 的子树中移除，加入根链表"""
    # 从 y 的子链表中移除 x
    if x.right == x:
        y.child = None
    else:
        if y.child == x:
            y.child = x.right
        x.left.right = x.right
        x.right.left = x.left

    y.degree -= 1
    x.parent = None
    x.mark = False
    self._add_to_root_list(x)

def _cascading_cut(self, y):
    """级联剪切"""
    parent = y.parent
    if parent is not None:
        if not y.mark:
            y.mark = True
        else:
            self._cut(y, parent)
            self._cascading_cut(parent)
```

## 4. 势能分析

### 4.1 势能函数

```
Phi(H) = t(H) + 2 * m(H)
```

其中：
- t(H) = 根链表中的节点数
- m(H) = 被标记的节点数

### 4.2 均摊复杂度推导

| 操作 | 实际代价 | 势能变化 | 均摊代价 |
|------|---------|---------|---------|
| 插入 | O(1) | +1 | O(1) |
| 合并 | O(1) | +0 | O(1) |
| 删除最小值 | O(D(n)+t) | -t+D(n) | O(D(n))=O(log n) |
| 减小键值 | O(c) | -c+2 | O(1) |

其中 D(n) = O(log n) 是最大度数上界。

## 5. 在 Dijkstra 算法中的应用

```python
def dijkstra_fib_heap(graph, start):
    """使用斐波那契堆优化的 Dijkstra"""
    dist = {v: float('inf') for v in graph}
    dist[start] = 0

    fh = FibonacciHeap()
    node_map = {}
    for v in graph:
        node_map[v] = fh.insert(0 if v == start else float('inf'), v)

    while not fh.is_empty():
        u_node = fh.extract_min()
        u = u_node.value

        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                fh.decrease_key(node_map[v], dist[v])

    return dist
```

## 6. 复杂度对比

| 操作 | 二叉堆 | 二项堆 | 斐波那契堆 |
|------|--------|--------|-----------|
| 插入 | O(log n) | O(log n) 均摊 | O(1) 均摊 |
| 合并 | O(n) | O(log n) | O(1) 均摊 |
| 删除最小值 | O(log n) | O(log n) | O(log n) 均摊 |
| 减小键值 | O(log n) | O(log n) | O(1) 均摊 |

## 7. 实际应用

| 算法 | 使用斐波那契堆的优势 |
|------|---------------------|
| Dijkstra | O(E + V log V) vs O((E+V) log V) |
| Prim | O(E + V log V) vs O((E+V) log V) |
| 最小生成树 | 稀疏图上更优 |

## 8. 总结

斐波那契堆在理论上具有最优的渐进复杂度，特别是 O(1) 的插入和减小键值操作。但在实际应用中，由于常数因子较大、实现复杂，通常只在稠密图的最短路径算法中才有实际优势。
