# B+树 (B+ Tree)

## 1. 概述

B+树是B树的一种变种，专门为数据库和文件系统优化。相比B树，B+树做了以下关键改变：
- 内部节点只存关键字，不存数据（数据全部存在叶节点）
- 叶节点之间用链表连接
- 所有查找最终都到达叶节点

## 2. B+树与B树的核心区别

| 特性 | B树 | B+树 |
|------|-----|------|
| 数据存储 | 所有节点 | 仅叶节点 |
| 内部节点 | 存关键字+数据 | 只存关键字+指针 |
| 叶节点连接 | 无 | 双向/单向链表 |
| 范围查询 | 需要中序遍历 | 沿叶节点链表扫描 |
| 复制关键字 | 不允许 | 内部节点可重复 |
| 空间利用率 | 较低 | 更高（内部节点更紧凑） |

## 3. B+树的节点结构

### 3.1 内部节点

```
[k1 | k2 | k3]
 /    |    |    \
p0   p1   p2   p3
```

只存储关键字和子节点指针，不存储实际数据记录。关键字是其右子树中最小关键字的副本。

### 3.2 叶节点

```
[k1 | k2 | k3] -> [data1, data2, data3] -> next指针
```

存储关键字和对应的数据（或数据指针），叶节点之间通过链表连接。

## 4. B+树节点定义

### 4.1 Python 实现

```python
class BPlusTreeNode:
    """B+树节点"""
    def __init__(self, t, leaf=False):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf
        self.next = None     # 叶节点链表指针
        self.n = 0

    def is_full(self):
        if self.leaf:
            return self.n >= 2 * self.t
        return self.n >= 2 * self.t - 1
```

## 5. 查找操作

### 5.1 精确查找

```python
def search(self, key):
    """精确查找关键字"""
    node = self.root

    while not node.leaf:
        i = 0
        while i < node.n and key >= node.keys[i]:
            i += 1
        node = node.children[i]

    for i in range(node.n):
        if node.keys[i] == key:
            return node.children[i]

    return None
```

### 5.2 范围查询（B+树的核心优势）

```python
def range_search(self, low, high):
    """范围查询：返回 [low, high] 范围内的所有记录"""
    # 先找到 >= low 的叶节点
    node = self.root
    while not node.leaf:
        i = 0
        while i < node.n and low >= node.keys[i]:
            i += 1
        node = node.children[i]

    # 沿叶节点链表扫描
    result = []
    while node is not None:
        for i in range(node.n):
            if node.keys[i] > high:
                return result
            if node.keys[i] >= low:
                result.append((node.keys[i], node.children[i]))
        node = node.next

    return result
```

### 5.3 为什么B+树适合范围查询

范围查询只需：
1. O(log_t n) 次I/O找到起始叶节点
2. O(k/t) 次I/O沿链表扫描结果

而B树需要完整的中序遍历，每次可能涉及不同的子树。

## 6. 插入操作

### 6.1 叶节点分裂

```python
def _split_leaf(self, leaf):
    """分裂叶节点"""
    mid = leaf.n // 2

    new_leaf = BPlusTreeNode(self.t, leaf=True)
    new_leaf.keys = leaf.keys[mid:]
    new_leaf.children = leaf.children[mid:]
    new_leaf.n = leaf.n - mid
    new_leaf.next = leaf.next  # 维护链表

    leaf.keys = leaf.keys[:mid]
    leaf.children = leaf.children[:mid]
    leaf.n = mid
    leaf.next = new_leaf

    return new_leaf.keys[0], new_leaf
```

### 6.2 完整插入代码

```python
def insert(self, key, value):
    """插入关键字和数据"""
    root = self.root

    if root.is_full():
        new_root = BPlusTreeNode(self.t, leaf=False)
        new_root.children.append(root)
        self._split_and_insert(new_root, 0, key, value)
        self.root = new_root
    else:
        self._insert_non_full(root, key, value)

def _insert_non_full(self, node, key, value):
    """在非满节点中插入"""
    if node.leaf:
        i = 0
        while i < node.n and key > node.keys[i]:
            i += 1
        node.keys.insert(i, key)
        node.children.insert(i, value)
        node.n += 1
    else:
        i = 0
        while i < node.n and key >= node.keys[i]:
            i += 1

        if node.children[i].is_full():
            split_key, new_child = self._split_child(node, i)
            if key >= split_key:
                i += 1
            self._insert_non_full(node.children[i], key, value)
        else:
            self._insert_non_full(node.children[i], key, value)
```

## 7. 删除操作

删除策略：
1. 找到包含关键字的叶节点
2. 从叶节点中删除
3. 如果叶节点关键字数低于下限，从兄弟借或合并
4. 更新父节点中的分隔关键字

B+树删除时，内部节点中可能保留关键字的副本，因为内部节点的关键字仅起路由作用。

## 8. B+树的度数设计

假设磁盘页大小 4KB：
- 每个关键字 8 字节
- 每个指针 8 字节
- 内部节点：约 256 个关键字
- 叶节点（假设数据记录200字节）：约 19 个记录

查询代价：
| 数据量 | B+树高度 | I/O次数 |
|--------|---------|---------|
| 100万 | 3 | 3次 |
| 10亿 | 4 | 4次 |
| 1万亿 | 5 | 5次 |

## 9. MySQL InnoDB中的B+树

- 聚簇索引：叶节点存储完整行数据
- 二级索引：叶节点存储主键值（需回表查询）
- 页大小：默认 16KB

## 10. 应用场景对比

| 场景 | B树 | B+树 |
|------|-----|------|
| 精确查找 | O(log_t n) | O(log_t n) |
| 范围查询 | O(n) 中序遍历 | O(log_t n + k) |
| 顺序访问 | 需要遍历 | 沿链表扫描 |
| 空间利用 | 数据分散 | 数据集中叶节点 |
| 内存利用 | 内部节点大 | 内部节点紧凑 |

## 11. 总结

B+树是数据库和文件系统的首选索引结构，核心优势：
1. 叶节点链表支持高效的范围查询和顺序访问
2. 内部节点紧凑（只存关键字），树高更低
3. 数据集中在叶节点，缓存更友好
4. 所有查询代价相同，性能可预测
