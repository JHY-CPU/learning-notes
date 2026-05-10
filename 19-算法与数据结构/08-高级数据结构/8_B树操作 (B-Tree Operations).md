# B树操作详解 (B-Tree Operations)

## 1. B树的核心操作

B树的三大核心操作：
1. 查找（Search）：在B树中定位关键字
2. 插入（Insert）：插入新关键字，可能触发节点分裂
3. 删除（Delete）：删除关键字，可能触发节点合并

## 2. 查找操作

### 2.1 非递归查找

```python
def search(self, key):
    """非递归查找，返回 (节点, 索引) 或 None"""
    node = self.root
    while node is not None:
        i = 0
        while i < node.n and key > node.keys[i]:
            i += 1

        if i < node.n and node.keys[i] == key:
            return (node, i)

        if node.leaf:
            return None

        node = node.children[i]

    return None
```

### 2.2 查找时间复杂度

- 每个节点内：O(t) 线性搜索 或 O(log t) 二分搜索
- 树的层数：O(log_t n)
- 总复杂度：O(t * log_t n)

## 3. 分裂操作（Split）

### 3.1 分裂原理

当一个节点满了（有 2t-1 个关键字）时，需要分裂：
1. 取中间关键字 kt 上移到父节点
2. 原节点分成左右两个，各有 t-1 个关键字
3. 原来的 2t 个子节点也分成两半

```
分裂前（t=2，满了3个关键字）:
  [10 | 20 | 30]

分裂后（中间关键字20上移）:
        [20]
       /    \
    [10]   [30]
```

### 3.2 分裂代码

```python
def _split_child(self, parent, i):
    """分裂 parent.children[i]"""
    t = self.t
    full_node = parent.children[i]

    # 创建新节点，保存右半部分
    new_node = BTreeNode(t, full_node.leaf)
    new_node.n = t - 1
    new_node.keys = full_node.keys[t:]

    # 如果不是叶节点，复制后 t 个子节点
    if not full_node.leaf:
        new_node.children = full_node.children[t:]

    # 中间关键字上移到父节点
    mid_key = full_node.keys[t - 1]

    # 在父节点中插入中间关键字
    parent.keys.insert(i, mid_key)
    parent.children.insert(i + 1, new_node)
    parent.n += 1

    # 截断原节点
    full_node.keys = full_node.keys[:t - 1]
    if not full_node.leaf:
        full_node.children = full_node.children[:t]
    full_node.n = t - 1
```

## 4. 插入操作

### 4.1 插入策略

B树的插入从叶节点开始：
1. 如果根节点满了，先分裂根节点
2. 递归向下找到合适的叶节点
3. 如果叶节点满了，先分裂再插入
4. 在叶节点中插入关键字

### 4.2 完整插入代码

```python
class BTree:
    def __init__(self, t):
        self.t = t
        self.root = BTreeNode(t, leaf=True)

    def insert(self, key):
        """插入新关键字"""
        root = self.root

        # 根节点满了，先分裂
        if root.is_full():
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
            self._insert_non_full(new_root, key)
        else:
            self._insert_non_full(root, key)

    def _insert_non_full(self, node, key):
        """在非满节点中插入关键字"""
        i = node.n - 1

        if node.leaf:
            # 叶节点：直接插入
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
            node.n += 1
        else:
            # 内部节点：找到合适的子节点
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key)
```

### 4.3 插入示例（t=2）

```
依次插入: 10, 20, 5, 6, 12

插入10:   [10]
插入20:   [10, 20]
插入5:    [5, 10, 20] <- 满了！
插入6:    先分裂 -> [10]
                     /   \
                  [5]    [20]
          再插入6 -> [10]
                    /    \
                 [5,6]   [20]
插入12:   [10]
         /    \
      [5,6]  [12,20]
```

## 5. 删除操作

### 5.1 删除策略

删除比插入复杂，分三种情况：

情况1：关键字在叶节点中
- 如果叶节点关键字数 >= t，直接删除
- 否则需要从兄弟节点借关键字或合并

情况2：关键字在内部节点中
- 如果左子节点关键字数 >= t，用前驱替代
- 如果右子节点关键字数 >= t，用后继替代
- 否则合并两个子节点，再递归删除

### 5.2 合并操作

```python
def _merge(self, node, idx):
    """将 node.children[idx+1] 合并到 node.children[idx]"""
    child = node.children[idx]
    sibling = node.children[idx + 1]

    # 将分隔关键字下移到左子节点
    child.keys.append(node.keys[idx])
    child.n += 1

    # 将右兄弟的关键字和子节点合并过来
    child.keys.extend(sibling.keys)
    child.children.extend(sibling.children)
    child.n += sibling.n

    # 从父节点移除分隔关键字和右兄弟指针
    node.keys.pop(idx)
    node.children.pop(idx + 1)
    node.n -= 1

    # 如果父节点变空，用合并后的子节点作为新根
    if node.n == 0 and node == self.root:
        self.root = child
```

### 5.3 从兄弟借关键字

```python
def _borrow_from_prev(self, node, idx):
    """从左兄弟借关键字"""
    child = node.children[idx]
    sibling = node.children[idx - 1]

    child.keys.insert(0, node.keys[idx - 1])
    child.n += 1

    node.keys[idx - 1] = sibling.keys.pop()
    sibling.n -= 1

    if not sibling.leaf:
        child.children.insert(0, sibling.children.pop())
```

## 6. 操作复杂度总结

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 查找 | O(t * log_t n) | 每层线性搜索 |
| 插入 | O(t * log_t n) | 含分裂开销 |
| 删除 | O(t * log_t n) | 含合并开销 |
| 分裂 | O(t) | 复制关键字 |
| 合并 | O(t) | 合并关键字 |
| 遍历 | O(n) | 访问所有关键字 |

## 7. 总结

B树的核心操作围绕分裂和合并展开：
- 插入时自顶向下，满节点先分裂
- 删除时处理叶节点不够的情况，通过借或合并
- 所有操作保证树高为 O(log_t n)
- 节点大小与磁盘页对齐，I/O效率极高
