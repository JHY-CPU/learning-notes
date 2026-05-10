# 树状数组基础 (Fenwick Tree Basics)

## 1. 概述

树状数组（Fenwick Tree / Binary Indexed Tree，简称BIT），由 Peter Fenwick 于 1994 年提出。它用非常巧妙的方式在数组上实现**前缀和的高效维护**，代码量远少于线段树。

核心操作：
- 单点修改：O(log n)
- 前缀和查询：O(log n)

## 2. lowbit 运算

### 2.1 定义

lowbit(x) 表示 x 的二进制表示中，最低位的 1 所对应的值。

```
lowbit(x) = x & (-x)  // 利用补码的性质
```

### 2.2 示例

| x (十进制) | x (二进制) | -x (二进制) | lowbit(x) |
|-----------|-----------|------------|-----------|
| 6 | 0110 | 1010 | 0010 = 2 |
| 8 | 1000 | 1000 | 1000 = 8 |
| 12 | 1100 | 0100 | 0100 = 4 |
| 7 | 0111 | 1001 | 0001 = 1 |

### 2.3 代码

```python
def lowbit(x):
    """返回x的最低位1对应的值"""
    return x & (-x)
```

```cpp
int lowbit(int x) {
    return x & (-x);
}
```

## 3. 树状数组的结构

### 3.1 数组表示

树状数组用一个数组 tree[] 维护，其中：
- tree[i] 存储原数组中一段区间的和
- tree[i] 管辖的区间为 [i - lowbit(i) + 1, i]

### 3.2 管辖范围

| i | 二进制 | lowbit(i) | 管辖区间 |
|---|--------|-----------|---------|
| 1 | 0001 | 1 | [1, 1] |
| 2 | 0010 | 2 | [1, 2] |
| 3 | 0011 | 1 | [3, 3] |
| 4 | 0100 | 4 | [1, 4] |
| 5 | 0101 | 1 | [5, 5] |
| 6 | 0110 | 2 | [5, 6] |
| 7 | 0111 | 1 | [7, 7] |
| 8 | 1000 | 8 | [1, 8] |

### 3.3 树形结构示意

```
          8
        / | \
       4  |  6
      /|\ | / \
     2 |  ||   7
    /\ |  ||
   1  3 5
```

- tree[8] 管辖 [1..8]
- tree[4] 管辖 [1..4]，tree[6] 管辖 [5..6]
- tree[2] 管辖 [1..2]，tree[7] 管辖 [7..7]

## 4. 前缀和查询

### 4.1 原理

查询前缀和 a[1] + a[2] + ... + a[idx] 时，沿 idx 不断减去 lowbit(idx)，累加经过的 tree 值。

```
query(7) = tree[7] + tree[6] + tree[4]
         = a[7] + (a[5]+a[6]) + (a[1]+a[2]+a[3]+a[4])
         = a[1]+a[2]+a[3]+a[4]+a[5]+a[6]+a[7]
```

### 4.2 代码

```python
def query(self, idx):
    """查询前缀和 [1, idx]"""
    result = 0
    while idx > 0:
        result += self.tree[idx]
        idx -= lowbit(idx)  # 去掉最低位1
    return result
```

```cpp
int query(int idx) {
    int result = 0;
    while (idx > 0) {
        result += tree[idx];
        idx -= lowbit(idx);
    }
    return result;
}
```

## 5. 单点修改

### 5.1 原理

将 a[idx] 增加 val 时，需要更新所有包含 idx 的 tree 值。沿 idx 不断加上 lowbit(idx)。

```
update(3, val):
  tree[3] += val  (3 = 0011)
  tree[4] += val  (4 = 0100, 3+lowbit(3)=4)
  tree[8] += val  (8 = 1000, 4+lowbit(4)=8)
```

### 5.2 代码

```python
def update(self, idx, val):
    """将 a[idx] 增加 val"""
    while idx <= self.n:
        self.tree[idx] += val
        idx += lowbit(idx)  # 加上最低位1
```

```cpp
void update(int idx, int val) {
    while (idx <= n) {
        tree[idx] += val;
        idx += lowbit(idx);
    }
}
```

## 6. 区间查询

利用前缀和的差来实现区间查询：

```python
def range_query(self, l, r):
    """查询区间 [l, r] 的和"""
    return self.query(r) - self.query(l - 1)
```

## 7. 时间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 单点修改 | O(log n) | 沿树向上更新 |
| 前缀和查询 | O(log n) | 沿树向下累加 |
| 区间查询 | O(log n) | 两次前缀和查询 |
| 建树 | O(n log n) | n次update |
| 建树（优化）| O(n) | 从后向前利用子节点 |

## 8. 与线段树对比

| 特性 | 树状数组 | 线段树 |
|------|---------|--------|
| 代码量 | 很少（~10行） | 较多（~50行） |
| 空间 | O(n) | O(4n) |
| 常数 | 非常小 | 较大 |
| 单点修改+前缀和 | 支持 | 支持 |
| 区间修改+区间查询 | 需技巧 | 懒标记直接支持 |
| 区间最值 | 不支持 | 支持 |
| 灵活性 | 有限 | 高 |

## 9. 应用场景

1. 动态前缀和维护
2. 逆序对计数
3. 区间计数（配合离散化）
4. 多维前缀和（二维BIT）
5. 作为其他算法的子模块

## 10. 总结

树状数组是一种轻量级的区间数据结构：
- 核心是 lowbit 运算实现的巧妙索引
- 代码极简，常数极小
- 适合单点修改+前缀和查询的场景
- 对于更复杂的区间操作，线段树更合适
