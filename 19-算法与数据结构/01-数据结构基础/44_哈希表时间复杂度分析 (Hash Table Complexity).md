# 45-哈希表时间复杂度分析 (Hash Table Complexity)

哈希表的性能取决于哈希函数质量、冲突解决策略和负载因子。平均情况下所有操作都是 O(1)，但最坏情况下可能退化到 O(n)。

## 复杂度详细分析

| 操作 | 平均 | 最坏 | 均摊 |
|------|------|------|------|
| 插入 | O(1) | O(n) | O(1) |
| 查找 | O(1) | O(n) | - |
| 删除 | O(1) | O(n) | - |
| 扩容 | - | O(n) | O(1)/元素 |
| 遍历 | O(n) | O(n) | - |

## 负载因子分析

```
负载因子 α = n / m（n = 元素数，m = 桶数）

链地址法期望查找长度：1 + α/2
开放地址法期望查找长度：1 / (1 - α)

α = 0.5 时：链地址 1.25 次探测，开放地址 2 次探测
α = 0.75 时：链地址 1.375 次探测，开放地址 4 次探测
α = 0.9 时：链地址 1.45 次探测，开放地址 10 次探测
```

## 均摊分析

```javascript
// 扩容的均摊分析
// 假设初始容量为 1，每次容量翻倍
// 扩容发生在第 1, 2, 4, 8, 16, ... 次插入时
// 总复制次数 = 1 + 2 + 4 + 8 + ... + n = 2n - 1
// 均摊每次插入 = (n + 2n - 1) / n ≈ 3 = O(1)
```

## 哈希函数质量的影响

```javascript
// 好的哈希函数：均匀分布
function goodHash(key, capacity) {
  let hash = 0;
  for (const ch of String(key)) {
    hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  }
  return hash % capacity;
}

// 差的哈希函数：所有键映射到同一桶
function badHash(key, capacity) {
  return 0; // 退化为链表
}
```

## 实际性能对比

```javascript
function benchmark() {
  const sizes = [1000, 10000, 100000];
  for (const n of sizes) {
    // 哈希表查找
    const map = new Map();
    for (let i = 0; i < n; i++) map.set(`key${i}`, i);
    let start = performance.now();
    for (let i = 0; i < n; i++) map.get(`key${i}`);
    console.log(`Map ${n}: ${(performance.now() - start).toFixed(2)}ms`);

    // 数组线性查找
    const arr = [];
    for (let i = 0; i < n; i++) arr.push([`key${i}`, i]);
    start = performance.now();
    for (let i = 0; i < n; i++) arr.find(p => p[0] === `key${i}`);
    console.log(`Array ${n}: ${(performance.now() - start).toFixed(2)}ms`);
  }
}
```

## 与其他数据结构对比

| 数据结构 | 查找 | 插入 | 删除 | 有序遍历 |
|---------|------|------|------|---------|
| 哈希表 | O(1)* | O(1)* | O(1)* | O(n log n) |
| 平衡BST | O(log n) | O(log n) | O(log n) | O(n) |
| 跳表 | O(log n)* | O(log n)* | O(log n)* | O(n) |
| 数组(无序) | O(n) | O(1) | O(n) | O(n log n) |
| 数组(有序) | O(log n) | O(n) | O(n) | O(n) |

*平均情况

## 性能调优建议

1. **预设容量**：如果知道数据量，初始化时设好容量避免频繁扩容
2. **负载因子**：查找密集场景用较低负载因子（0.5），插入密集用较高（0.75）
3. **哈希函数**：大表用 MurmurHash、CityHash 等高质量哈希函数
4. **冲突解决**：数据量大时链地址法 + 红黑树优化（Java 8+ HashMap）

## 常见陷阱

1. **最坏情况不等于平均**：面试中要能分析最坏情况
2. **均摊分析**：扩容的单次 O(n) 操作均摊后是 O(1)
3. **JavaScript 引擎优化**：大量键时引擎会改变内部结构
4. **安全问题**：哈希碰撞攻击（HashDoS）可导致拒绝服务
