# 海量数据处理 (Big Data Processing)

## 一、核心思想

海量数据处理的核心：数据量太大无法一次性装入内存时，使用**分治、哈希、位图、堆**等技巧来处理。

### 1.1 数据量与内存的关系

| 数据量 | 内存占用 | 处理策略 |
|--------|---------|---------|
| 10^6 个int | ~4MB | 直接处理 |
| 10^8 个int | ~400MB | 可能需要优化 |
| 10^9 个int | ~4GB | 必须分治处理 |
| 10^10 个int | ~40GB | 外部排序、哈希分片 |

---

## 二、核心技巧

### 2.1 哈希分片 (Hash Partitioning)

将大数据分成多个小文件分别处理。

```python
# 10亿个数中找出现次数超过 N/10 的数
# 思路：哈希分片后分别计数

def find_frequent(nums, k=10):
    from collections import Counter

    # 分片
    partitions = [Counter() for _ in range(k)]
    for num in nums:
        partitions[num % k][num] += 1

    # 每个分片独立找
    candidates = Counter()
    for part in partitions:
        for num, cnt in part.items():
            candidates[num] += cnt

    return [num for num, cnt in candidates.items() if cnt > len(nums) // k]
```

### 2.2 位图法 (BitMap)

适用于判断元素是否存在、去重、排序。

```python
# 40亿个无符号整数中找不存在的数
# 用 512MB 位图（2^32 bit ≈ 512MB）

def find_missing(nums, max_val=2**32):
    bitmap = bytearray(max_val // 8 + 1)

    for num in nums:
        bitmap[num >> 3] |= (1 << (num & 7))

    for i in range(max_val):
        if not (bitmap[i >> 3] & (1 << (i & 7))):
            return i
    return -1
```

### 2.3 堆 (TopK问题)

```python
import heapq

# 10亿个数中找最大的100个
def top_k_large(nums, k):
    # 最小堆，保持堆大小为k
    heap = []
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)
    return sorted(heap, reverse=True)

# 找最小的100个 — 用最大堆
def top_k_small(nums, k):
    heap = []
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, -num)
        elif num < -heap[0]:
            heapq.heapreplace(heap, -num)
    return sorted([-x for x in heap])
```

### 2.4 布隆过滤器

用于判断某元素是否在集合中（允许假阳性，不允许假阴性）。

```python
# 场景：10亿个URL的黑名单，判断某个URL是否在其中
# 布隆过滤器：1GB位数组 + 7个哈希函数

import mmh3

class BloomFilter:
    def __init__(self, size=8*10**9, hash_count=7):
        self.size = size
        self.hash_count = hash_count
        self.bits = bytearray(size // 8 + 1)

    def add(self, item):
        for seed in range(self.hash_count):
            idx = mmh3.hash(str(item), seed) % self.size
            self.bits[idx >> 3] |= (1 << (idx & 7))

    def contains(self, item):
        for seed in range(self.hash_count):
            idx = mmh3.hash(str(item), seed) % self.size
            if not (self.bits[idx >> 3] & (1 << (idx & 7))):
                return False
        return True
```

---

## 三、经典面试题

### 3.1 两个大文件的交集

**场景：** 两个文件各含50亿个URL，找出共同的URL。

**方案：**
1. 分别哈希到1000个小文件
2. 对相同编号的小文件求交集
3. 用HashSet判断

### 3.2 10亿个数中找中位数

**思路：** 分段统计。将值域分成若干区间，先确定中位数在哪个区间，再在该区间内精确查找。

```python
def find_median_stream(stream, total_count):
    # 分成 2^16 个桶
    num_buckets = 65536
    buckets = [0] * num_buckets
    max_val = 2**32

    for num in stream:
        buckets[num // (max_val // num_buckets)] += 1

    # 找中位数所在的桶
    target = total_count // 2
    cumulative = 0
    for i, cnt in enumerate(buckets):
        cumulative += cnt
        if cumulative > target:
            return i  # 返回桶编号
```

### 3.3 找出出现次数最多的前K个

```python
# 哈希分片 + 堆
def top_k_frequent_huge(data_stream, k, num_partitions=100):
    import heapq
    from collections import Counter

    # 哈希分片
    partitions = [Counter() for _ in range(num_partitions)]
    for item in data_stream:
        partitions[hash(item) % num_partitions][item] += 1

    # 每个分片维护top-k
    global_counter = Counter()
    for part in partitions:
        global_counter.update(part)

    return heapq.nlargest(k, global_counter.keys(), key=global_counter.get)
```

---

## 四、外部排序

### 4.1 外部归并排序

```
1. 将大文件分成若干能装入内存的小段
2. 对每段在内存中排序，写回磁盘
3. 多路归并合并所有有序段
```

```python
import heapq

def external_merge_sort(input_file, output_file, chunk_size=10**6):
    # 步骤1：分段排序
    chunks = []
    with open(input_file) as f:
        while True:
            lines = [f.readline() for _ in range(chunk_size)]
            lines = [l for l in lines if l]
            if not lines: break
            lines.sort()
            chunk_file = f"chunk_{len(chunks)}.txt"
            with open(chunk_file, 'w') as cf:
                cf.writelines(lines)
            chunks.append(chunk_file)

    # 步骤2：多路归并
    files = [open(f) for f in chunks]
    heap = []
    for i, f in enumerate(files):
        line = f.readline()
        if line:
            heapq.heappush(heap, (line, i))

    with open(output_file, 'w') as out:
        while heap:
            line, i = heapq.heappop(heap)
            out.write(line)
            next_line = files[i].readline()
            if next_line:
                heapq.heappush(heap, (next_line, i))

    for f in files: f.close()
```

---

## 五、复杂度分析

| 方法 | 时间 | 空间 | 适用场景 |
|------|------|------|---------|
| 哈希分片 | $O(n)$ | $O(n/k)$ | 频率统计 |
| 位图 | $O(n)$ | $O(max/8)$ | 存在性判断 |
| 堆 | $O(n \log k)$ | $O(k)$ | TopK |
| 布隆过滤器 | $O(n)$ | $O(m/8)$ | 黑名单 |
| 外部排序 | $O(n \log n)$ | $O(chunk)$ | 全量排序 |
