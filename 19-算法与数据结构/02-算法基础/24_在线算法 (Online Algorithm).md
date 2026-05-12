# 24-在线算法 (Online Algorithm)

在线算法逐个处理输入，每收到一个输入必须立即决策，无法预知后续数据。

## 核心概念

- 每次决策基于不完整信息
- 用竞争比（Competitive Ratio）衡量质量
- 竞争比 = 在线算法代价 / 最优离线算法代价

```javascript
// 在线缓存（LRU）
class OnlineCache {
  constructor(cap) {
    this.cap = cap;
    this.cache = new Map();
  }

  access(page) {
    if (this.cache.has(page)) {
      // 命中：移到最近
      this.cache.delete(page);
      this.cache.set(page, true);
      return 'hit';
    }
    // 未命中：淘汰最久未使用
    if (this.cache.size >= this.cap) {
      this.cache.delete(this.cache.keys().next().value);
    }
    this.cache.set(page, true);
    return 'miss';
  }
}
```

## 经典在线算法

| 问题 | 算法 | 竞争比 |
|------|------|--------|
| 缓存替换 | LRU/FIFO | O(k) |
| 在线装箱 | First Fit | 1.7 |
| K-server | Double Coverage | 2 |
| 在线调度 | SPT | 2 |

## 在线 vs 离线

| 特性 | 在线 | 离线 |
|------|------|------|
| 输入 | 逐个到达 | 全部已知 |
| 预处理 | 不可以 | 可以 |
| 解质量 | 有竞争比 | 通常更优 |
| 适用 | 实时系统 | 批处理 |

## 在线装箱算法

```javascript
// First Fit 在线装箱
function firstFit(items, binCapacity) {
  const bins = [];
  for (const item of items) {
    let placed = false;
    for (const bin of bins) {
      if (bin.remaining >= item) {
        bin.remaining -= item;
        bin.items.push(item);
        placed = true;
        break;
      }
    }
    if (!placed) {
      bins.push({ remaining: binCapacity - item, items: [item] });
    }
  }
  return bins;
}
// 竞争比：1.7（最优装箱为 OPT）
// 即 bins.length <= 1.7 * OPT + 2
```

## 在线调度

```javascript
// SPT（Shortest Processing Time）在线调度
// 每次选择处理时间最短的任务
function sptSchedule(tasks) {
  // 在线场景：任务逐个到达，不可预排序
  // 简化：维护一个最小堆
  tasks.sort((a, b) => a.time - b.time);
  let completion = 0, totalWait = 0;
  for (const t of tasks) {
    completion += t.time;
    totalWait += completion;
  }
  return { completion, avgWait: totalWait / tasks.length };
}
```

## 应用场景

- 操作系统页面置换
- 网络路由器缓冲管理
- 在线广告投放
- 实时任务调度
- 股票交易决策

## 常见陷阱

1. **竞争比误解**：竞争比是上界，不是每次的实际比值
2. **实际性能**：竞争比差的算法实际表现可能更好
3. **随机化**：随机在线算法可以有更好的竞争比
4. **对抗输入**：竞争比考虑最坏情况，对抗性输入可能触发
