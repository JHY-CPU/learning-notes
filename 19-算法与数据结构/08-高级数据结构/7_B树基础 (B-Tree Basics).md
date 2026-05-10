# B树基础 (B-Tree Basics)

## 1. 概述

B树是一种**自平衡的多路搜索树**，由 Rudolf Bayer 和 Edward McCreight 于 1972 年提出。B树专门为磁盘等外部存储设计，能够最小化磁盘I/O操作次数。

B树广泛应用于：
- 数据库索引（MySQL InnoDB、PostgreSQL）
- 文件系统（NTFS、ext4、HFS+）
- 键值存储系统

## 2. m阶B树的定义

### 2.1 节点结构

每个节点包含：
- 关键字（keys）：按升序排列，最多 m-1 个
- 子节点指针（children）：最多 m 个
- 关键字 ki 与子节点指针 pi 交替排列

### 2.2 五条约束条件

| 约束 | 内容 |
|------|------|
| 1. 根节点 | 至少有 2 个子节点（除非树只有根节点） |
| 2. 内部节点 | 非叶非根节点至少有 ceil(m/2) 个子节点 |
| 3. 所有叶子 | 位于同一层（完美平衡） |
| 4. 关键字数 | 内部节点有 [ceil(m/2)-1, m-1] 个关键字 |
| 5. 子节点数 | 有 n 个关键字的节点有 n+1 个子节点 |

### 2.3 以3阶B树为例

```
       [10 | 20]
      /    |    \
  [3|5]  [15]  [25|30|35]
```

根节点有2个关键字，3个子节点。所有叶子在同一层。

## 3. B树节点定义

### 3.1 Python 实现

```python
class BTreeNode:
    """B树节点"""
    def __init__(self, t, leaf=False):
        self.t = t              # 最小度数（minimum degree）
        self.keys = []          # 关键字列表
        self.children = []      # 子节点指针列表
        self.leaf = leaf        # 是否为叶节点
        self.n = 0              # 当前关键字个数

    def is_full(self):
        """节点是否已满（有 2t-1 个关键字）"""
        return self.n == 2 * self.t - 1
```

### 3.2 C++ 实现

```cpp
template <int T>
struct BTreeNode {
    int keys[2 * T - 1];
    BTreeNode* children[2 * T];
    int n;
    bool leaf;

    BTreeNode(bool isLeaf) : n(0), leaf(isLeaf) {}
};
```

### 3.3 最小度数 t 与阶 m 的关系

- 每个节点（除根外）关键字数：t-1 到 2t-1
- 每个节点子节点数：t 到 2t
- 阶 m = 2t

## 4. B树的查找操作

```python
def search(self, node, key):
    """在B树中查找关键字"""
    i = 0
    while i < node.n and key > node.keys[i]:
        i += 1

    if i < node.n and node.keys[i] == key:
        return (node, i)

    if node.leaf:
        return None

    return self.search(node.children[i], key)
```

查找复杂度：O(t * log_t n) 或使用二分搜索 O(log n)。

## 5. B树的高度

定理：含有 n 个关键字的 m阶B树，高度 h 满足：
h <= log_t((n+1)/2)

其中 t = ceil(m/2)。

以 t=128 为例，10亿条记录只需约3层。

## 6. B树与二叉搜索树对比

| 特性 | BST | B树 |
|------|-----|-----|
| 节点关键字数 | 1 | 最多 m-1 个 |
| 子节点数 | 2 | 最多 m 个 |
| 树高 | O(log n) | O(log_t n)，更低 |
| 磁盘I/O | 每层1次 | 每层1次，层数更少 |
| 适用场景 | 内存 | 磁盘/外存 |

## 7. B树的度数选择

实际应用中，B树的节点大小通常设为磁盘页大小（4KB）。

假设每个关键字 8 字节，指针 8 字节：
- 节点可容纳约 4096/16 = 256 个关键字
- t 约为 128，m 约为 256
- 10亿条记录只需约 3 层

为什么B树适合磁盘：
1. 节点大小与页对齐：一次I/O读取整个节点
2. 树高很低：减少I/O次数
3. 局部性好：同一节点内数据连续存储

## 8. B树的遍历

```python
def traverse(self, node):
    """中序遍历B树"""
    result = []
    for i in range(node.n):
        if not node.leaf:
            result.extend(self.traverse(node.children[i]))
        result.append(node.keys[i])
    if not node.leaf:
        result.extend(self.traverse(node.children[node.n]))
    return result
```

## 9. 应用场景

| 场景 | 说明 |
|------|------|
| 数据库索引 | MySQL InnoDB使用B+树变种 |
| 文件系统 | 管理文件目录和数据块 |
| 键值存储 | 有序存储和范围查询 |
| 外部排序 | 多路归并 |

## 10. 总结

B树通过多路分支大幅降低了树的高度，从而减少磁盘I/O次数。其核心特点：
- 每个节点存储多个关键字
- 所有叶节点在同一层
- 节点分裂和合并保证平衡
- 特别适合外部存储场景
