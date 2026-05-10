# 数据库索引模拟 (DB Index Simulator)

## 项目需求与功能分析

数据库索引是提升查询性能的关键技术。本项目模拟 B+ 树索引的增删查操作，帮助深入理解数据库索引的底层实现。

### 核心功能

- B+ 树的插入、查找、删除操作
- 节点分裂与合并的可视化
- 范围查询支持
- 与顺序扫描的性能对比
- B+ 树结构可视化

### B+ 树特性

- 所有数据存储在叶子节点
- 叶子节点通过链表连接（支持范围扫描）
- 非叶子节点仅存储键值和子节点指针
- 所有叶子节点在同一层（平衡树）
- 每个节点的键数在 [ceil(m/2)-1, m-1] 之间

## 核心算法原理

### 节点分裂

当节点键数超过 m-1 时，取中间键上提到父节点，节点一分为二。

### 节点合并

当节点键数低于 ceil(m/2)-1 时，从兄弟节点借一个键或与兄弟节点合并。

### 查找

从根节点开始，根据键值比较选择子节点，直到叶子节点。

## 完整代码实现

```python
import math
from typing import List, Tuple, Optional, Any


class BPlusTreeNode:
    """B+ 树节点"""

    def __init__(self, leaf=False, order=4):
        self.leaf = leaf
        self.keys: List[int] = []
        self.children: List[Any] = []  # 非叶子存子节点，叶子存数据
        self.next: Optional['BPlusTreeNode'] = None  # 叶子链表
        self.order = order

    def is_full(self):
        return len(self.keys) >= self.order - 1

    def min_keys(self):
        return math.ceil(self.order / 2) - 1


class BPlusTree:
    """B+ 树实现"""

    def __init__(self, order=4):
        self.order = order
        self.root = BPlusTreeNode(leaf=True, order=order)

    def search(self, key: int) -> Optional[Any]:
        """精确查找"""
        node = self.root
        while not node.leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        # 在叶子节点中查找
        for i, k in enumerate(node.keys):
            if k == key:
                return node.children[i]
        return None

    def range_search(self, low: int, high: int) -> List[Tuple[int, Any]]:
        """范围查询"""
        # 找到 low 所在的叶子节点
        node = self.root
        while not node.leaf:
            i = 0
            while i < len(node.keys) and low >= node.keys[i]:
                i += 1
            node = node.children[i]

        result = []
        while node:
            for i, k in enumerate(node.keys):
                if k > high:
                    return result
                if k >= low:
                    result.append((k, node.children[i]))
            node = node.next
        return result

    def insert(self, key: int, value: Any = None):
        """插入键值对"""
        root = self.root
        if root.is_full():
            # 根节点分裂
            new_root = BPlusTreeNode(order=self.order)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key, value)

    def _insert_non_full(self, node, key, value):
        if node.leaf:
            # 找到插入位置
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            node.keys.insert(i, key)
            node.children.insert(i, value)
        else:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, value)

    def _split_child(self, parent, index):
        node = parent.children[index]
        mid = len(node.keys) // 2

        # 创建新节点
        new_node = BPlusTreeNode(leaf=node.leaf, order=self.order)

        if node.leaf:
            # 叶子节点分裂
            new_node.keys = node.keys[mid:]
            new_node.children = node.children[mid:]
            node.keys = node.keys[:mid]
            node.children = node.children[:mid]
            # 维护叶子链表
            new_node.next = node.next
            node.next = new_node
            # 中间键提升（使用新节点的第一个键）
            parent.keys.insert(index, new_node.keys[0])
            parent.children.insert(index + 1, new_node)
        else:
            # 非叶子节点分裂
            mid_key = node.keys[mid]
            new_node.keys = node.keys[mid + 1:]
            new_node.children = node.children[mid + 1:]
            node.keys = node.keys[:mid]
            node.children = node.children[:mid + 1]
            parent.keys.insert(index, mid_key)
            parent.children.insert(index + 1, new_node)

    def delete(self, key: int):
        """删除键"""
        self._delete(self.root, key)
        # 如果根节点为空，降低树高
        if not self.root.keys and not self.root.leaf:
            self.root = self.root.children[0]

    def _delete(self, node, key):
        if node.leaf:
            if key in node.keys:
                idx = node.keys.index(key)
                node.keys.pop(idx)
                node.children.pop(idx)
            return

        # 找到子节点
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        child = node.children[i]

        if len(child.keys) <= child.min_keys():
            self._rebalance(node, i)

        # 重新定位
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        self._delete(node.children[i], key)

    def _rebalance(self, parent, index):
        child = parent.children[index]
        # 尝试从左兄弟借
        if index > 0:
            left = parent.children[index - 1]
            if len(left.keys) > left.min_keys():
                if child.leaf:
                    child.keys.insert(0, left.keys.pop())
                    child.children.insert(0, left.children.pop())
                    parent.keys[index - 1] = child.keys[0]
                return
        # 尝试从右兄弟借
        if index < len(parent.children) - 1:
            right = parent.children[index + 1]
            if len(right.keys) > right.min_keys():
                if child.leaf:
                    child.keys.append(right.keys.pop(0))
                    child.children.append(right.children.pop(0))
                    parent.keys[index] = right.keys[0]
                return
        # 合并
        if index > 0:
            left = parent.children[index - 1]
            left.keys.extend(child.keys)
            left.children.extend(child.children)
            if child.leaf:
                left.next = child.next
            parent.keys.pop(index - 1)
            parent.children.pop(index)
        elif index < len(parent.children) - 1:
            right = parent.children[index + 1]
            child.keys.extend(right.keys)
            child.children.extend(right.children)
            if right.leaf:
                child.next = right.next
            parent.keys.pop(index)
            parent.children.pop(index + 1)

    def display(self, node=None, level=0):
        """可视化树结构"""
        if node is None:
            node = self.root
        indent = "  " * level
        label = "叶" if node.leaf else "内"
        print(f"{indent}[{label}] 键: {node.keys}")
        if not node.leaf:
            for child in node.children:
                self.display(child, level + 1)

    def all_keys(self) -> List[int]:
        """获取所有键（有序）"""
        node = self.root
        while not node.leaf:
            node = node.children[0]
        result = []
        while node:
            result.extend(node.keys)
            node = node.next
        return result
```

## 测试用例

```python
import unittest

class TestBPlusTree(unittest.TestCase):
    def test_insert_search(self):
        tree = BPlusTree(order=4)
        for k, v in [(10,'a'),(20,'b'),(5,'c'),(6,'d'),(12,'e'),(30,'f')]:
            tree.insert(k, v)
        self.assertEqual(tree.search(10), 'a')
        self.assertEqual(tree.search(5), 'c')
        self.assertIsNone(tree.search(99))

    def test_range_search(self):
        tree = BPlusTree(order=4)
        for k in [5,10,15,20,25,30]: tree.insert(k, f"v{k}")
        result = tree.range_search(10, 25)
        keys = [k for k,_ in result]
        self.assertEqual(keys, [10,15,20,25])

    def test_sorted_order(self):
        tree = BPlusTree(order=3)
        import random
        keys = random.sample(range(100), 30)
        for k in keys: tree.insert(k)
        self.assertEqual(tree.all_keys(), sorted(keys))

    def test_delete(self):
        tree = BPlusTree(order=4)
        for k in [1,2,3,4,5,6,7,8]: tree.insert(k)
        tree.delete(5)
        self.assertIsNone(tree.search(5))
        self.assertEqual(tree.all_keys(), [1,2,3,4,6,7,8])

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **并发控制**：实现 B+ 树的锁机制（ crabbing 协议）
2. **持久化**：将 B+ 树节点存储到磁盘页
3. **复合索引**：支持多列联合索引
4. **前缀压缩**：非叶子节点中压缩公共前缀
5. **LSM 树**：实现 Log-Structured Merge Tree
6. **布隆过滤器**：加速不存在键的判断
7. **可视化 GUI**：图形化展示 B+ 树的分裂和合并过程
