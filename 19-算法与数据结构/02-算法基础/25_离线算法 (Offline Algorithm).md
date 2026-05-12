# 25-离线算法 (Offline Algorithm)

离线算法预先知道所有输入后统一处理，通常能取得更好的全局解。

## 核心优势

- 可以预处理数据（排序、建索引）
- 可以按最优顺序处理查询
- 通常比在线算法得到更好的解

```javascript
// 离线区间查询：前缀和
class OfflineRangeSum {
  constructor(arr) {
    this.prefix = [0];
    for (const x of arr) this.prefix.push(this.prefix.at(-1) + x);
  }
  query(l, r) { return this.prefix[r + 1] - this.prefix[l]; }
}

// 离线查询排序（莫队算法思路）
function offlineQueries(arr, queries) {
  // 按左端点分块，右端点排序
  const blockSize = Math.floor(Math.sqrt(arr.length));
  queries.sort((a, b) => {
    const ba = Math.floor(a[0] / blockSize), bb = Math.floor(b[0] / blockSize);
    if (ba !== bb) return ba - bb;
    return a[1] - b[1];
  });

  // 用前缀和回答每个查询
  const prefix = [0];
  for (const x of arr) prefix.push(prefix.at(-1) + x);
  return queries.map(([l, r]) => prefix[r + 1] - prefix[l]);
}
```

## 经典技巧

| 技巧 | 说明 | 复杂度 |
|------|------|--------|
| 前缀和 | 预处理后 O(1) 查询 | O(n) 预处理 |
| 差分数组 | 高效区间修改 | O(1) 修改 |
| 莫队算法 | 排序查询批量处理 | O(n*sqrt(n)) |
| CDQ 分治 | 高维偏序降维 | O(n log^2 n) |

## 离线 vs 在线

```
离线：已知所有输入 -> 预处理 -> 按最优顺序处理 -> 更好的解
在线：逐个输入 -> 立即决策 -> 无法回退 -> 竞争比保证
```

## 莫队算法详解

```javascript
// 莫队算法：离线区间查询
function moAlgorithm(arr, queries) {
  const n = arr.length;
  const blockSize = Math.floor(Math.sqrt(n));

  // 按块排序
  const sortedQ = queries.map((q, i) => ({...q, idx: i}))
    .sort((a, b) => {
      const ba = Math.floor(a.l / blockSize), bb = Math.floor(b.l / blockSize);
      if (ba !== bb) return ba - bb;
      return ba % 2 === 0 ? a.r - b.r : b.r - a.r;
    });

  const ans = new Array(queries.length);
  let curL = 0, curR = -1, curAns = 0;

  function add(pos) { curAns += arr[pos]; }
  function remove(pos) { curAns -= arr[pos]; }

  for (const q of sortedQ) {
    while (curR < q.r) add(++curR);
    while (curR > q.r) remove(curR--);
    while (curL < q.l) remove(curL++);
    while (curL > q.l) add(--curL);
    ans[q.idx] = curAns;
  }
  return ans;
}
```

## 差分数组

```javascript
// 差分数组：O(1) 区间修改，O(n) 最终查询
function diffArray(arr) {
  const n = arr.length;
  const diff = new Array(n + 1).fill(0);
  diff[0] = arr[0];
  for (let i = 1; i < n; i++) diff[i] = arr[i] - arr[i - 1];

  // 区间 [l, r] 加上 val
  function update(l, r, val) {
    diff[l] += val;
    diff[r + 1] -= val;
  }

  // 还原数组
  function build() {
    const result = [diff[0]];
    for (let i = 1; i < n; i++) result.push(result[i-1] + diff[i]);
    return result;
  }

  return { diff, update, build };
}
```

## 应用场景

- 数据库查询优化（重排执行顺序）
- 竞赛中莫队算法
- 批量数据处理
- 离线缓存预热

## 常见陷阱

1. **内存限制**：存储所有输入可能占用大量内存
2. **实时性**：不适用于需要实时响应的场景
3. **排序开销**：预处理的排序可能本身就很耗时
4. **在线转离线**：有时需要额外数据结构缓冲输入
