# 69-LFU缓存设计 (LFU Cache Design)

LFU（Least Frequently Used）缓存淘汰最不经常使用的数据，当频率相同时淘汰最久未使用的。

## 实现原理

- **cache**: key -> {val, freq} 存储值和频率
- **freqMap**: freq -> Set(key) 按频率分组
- **minFreq**: 跟踪当前最小频率
- get/put 时更新频率，淘汰时取 minFreq 组中最早加入的

## JavaScript 完整实现

```javascript
class LFUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = new Map();       // key -> {val, freq}
    this.freqMap = new Map();     // freq -> LinkedHashSet (用 Map 模拟有序)
    this.minFreq = 0;
  }

  _updateFreq(key) {
    const { val, freq } = this.cache.get(key);
    // 从旧频率组移除
    const oldSet = this.freqMap.get(freq);
    oldSet.delete(key);
    if (oldSet.size === 0) {
      this.freqMap.delete(freq);
      if (this.minFreq === freq) this.minFreq++;
    }
    // 添加到新频率组
    this.cache.set(key, { val, freq: freq + 1 });
    if (!this.freqMap.has(freq + 1)) this.freqMap.set(freq + 1, new Map());
    this.freqMap.get(freq + 1).set(key, true);
  }

  get(key) {
    if (!this.cache.has(key)) return -1;
    this._updateFreq(key);
    return this.cache.get(key).val;
  }

  put(key, value) {
    if (this.capacity === 0) return;
    if (this.cache.has(key)) {
      this.cache.get(key).val = value;
      this._updateFreq(key);
      return;
    }
    // 淘汰
    if (this.cache.size >= this.capacity) {
      const minSet = this.freqMap.get(this.minFreq);
      const evictKey = minSet.keys().next().value;
      minSet.delete(evictKey);
      if (minSet.size === 0) this.freqMap.delete(this.minFreq);
      this.cache.delete(evictKey);
    }
    // 插入新元素
    this.cache.set(key, { val, freq: 1 });
    if (!this.freqMap.has(1)) this.freqMap.set(1, new Map());
    this.freqMap.get(1).set(key, true);
    this.minFreq = 1;
  }
}

// 使用
const cache = new LFUCache(2);
cache.put(1, 1);
cache.put(2, 2);
cache.get(1);    // 1, freq[1]=2
cache.put(3, 3); // 淘汰 key=2 (freq=1)
cache.get(2);    // -1
cache.get(3);    // 3
```

## C++ 实现

```cpp
#include <unordered_map>
#include <list>
using namespace std;

class LFUCache {
    int cap, minFreq;
    unordered_map<int, pair<int,int>> cache;  // key -> {val, freq}
    unordered_map<int, list<int>> freqList;   // freq -> keys (LRU order)
    unordered_map<int, list<int>::iterator> pos; // key -> position

    void update(int key) {
        int freq = cache[key].second;
        freqList[freq].erase(pos[key]);
        if (freqList[freq].empty()) {
            freqList.erase(freq);
            if (minFreq == freq) minFreq++;
        }
        cache[key].second = freq + 1;
        freqList[freq + 1].push_front(key);
        pos[key] = freqList[freq + 1].begin();
    }

public:
    LFUCache(int capacity) : cap(capacity), minFreq(0) {}

    int get(int key) {
        if (!cache.count(key)) return -1;
        update(key);
        return cache[key].first;
    }

    void put(int key, int value) {
        if (cap <= 0) return;
        if (cache.count(key)) {
            cache[key].first = value;
            update(key);
            return;
        }
        if ((int)cache.size() >= cap) {
            int evict = freqList[minFreq].back();
            freqList[minFreq].pop_back();
            pos.erase(evict);
            cache.erase(evict);
        }
        cache[key] = {value, 1};
        freqList[1].push_front(key);
        pos[key] = freqList[1].begin();
        minFreq = 1;
    }
};
```

## LRU vs LFU

| 特性 | LRU | LFU |
|------|-----|-----|
| 淘汰策略 | 最久未使用 | 使用频率最低 |
| 实现复杂度 | 较简单 | 较复杂 |
| 适合场景 | 访问模式变化快 | 访问模式稳定 |
| 突发流量 | 适应好 | 适应差 |
| 历史记录 | 只看最近 | 累计频率 |

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| get | O(1) | - |
| put | O(1) | O(capacity) |

## 常见陷阱

1. **minFreq 更新**：淘汰元素后要正确更新 minFreq
2. **频率相同**：频率相同时按 LRU 策略淘汰（用有序结构）
3. **容量为 0**：特殊处理空缓存
4. **更新值**：put 已存在的 key 只更新值，不增加频率（除非同时 get）
