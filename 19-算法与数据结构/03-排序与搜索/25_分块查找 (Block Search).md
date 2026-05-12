# 26-分块查找 (Block Search)

分块查找（索引顺序查找）将数据分成若干块，块内无序但块间有序，先在索引表中定位块，再在块内顺序查找。

## 复杂度分析

| 指标 | 值 |
|------|-----|
| 平均时间 | O(sqrt(n)) |
| 最坏 | O(n) |
| 空间 | O(sqrt(n)) |

## JavaScript 实现

```javascript
// 分块查找
class BlockSearch {
  constructor(arr, blockSize) {
    this.arr = arr;
    this.blockSize = blockSize || Math.floor(Math.sqrt(arr.length));
    this.blocks = this._buildIndex();
  }

  _buildIndex() {
    const blocks = [];
    for (let i = 0; i < this.arr.length; i += this.blockSize) {
      const end = Math.min(i + this.blockSize, this.arr.length);
      const block = this.arr.slice(i, end);
      blocks.push({
        max: Math.max(...block),
        min: Math.min(...block),
        start: i,
        end: end - 1
      });
    }
    return blocks;
  }

  search(target) {
    // 1. 在索引表中定位块
    let blockIdx = -1;
    for (let i = 0; i < this.blocks.length; i++) {
      if (target >= this.blocks[i].min && target <= this.blocks[i].max) {
        blockIdx = i;
        break;
      }
    }
    if (blockIdx === -1) return -1;

    // 2. 在块内顺序查找
    const { start, end } = this.blocks[blockIdx];
    for (let i = start; i <= end; i++) {
      if (this.arr[i] === target) return i;
    }
    return -1;
  }

  // 插入元素
  insert(val) {
    this.arr.push(val);
    // 简化：重建索引
    this.blocks = this._buildIndex();
  }
}

// 测试
const data = [15, 20, 25, 10, 22, 30, 35, 40, 48, 32, 50, 60, 70, 55, 65];
const bs = new BlockSearch(data, 5);
console.log(bs.search(35));  // 6
console.log(bs.search(33));  // -1
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

struct Block {
    int maxVal, minVal, start, end;
};

class BlockSearch {
    vector<int>& arr;
    vector<Block> blocks;
    int blockSize;
public:
    BlockSearch(vector<int>& a, int bs) : arr(a), blockSize(bs) {
        buildIndex();
    }

    void buildIndex() {
        blocks.clear();
        for (int i = 0; i < arr.size(); i += blockSize) {
            int end = min(i + blockSize, (int)arr.size());
            int mx = *max_element(arr.begin() + i, arr.begin() + end);
            int mn = *min_element(arr.begin() + i, arr.begin() + end);
            blocks.push_back({mx, mn, i, end - 1});
        }
    }

    int search(int target) {
        for (int i = 0; i < blocks.size(); i++) {
            if (target >= blocks[i].minVal && target <= blocks[i].maxVal) {
                for (int j = blocks[i].start; j <= blocks[i].end; j++)
                    if (arr[j] == target) return j;
                return -1;
            }
        }
        return -1;
    }
};
```

## 与二分查找对比

| 特性 | 二分查找 | 分块查找 |
|------|---------|---------|
| 数据要求 | 完全有序 | 块间有序 |
| 时间 | O(log n) | O(sqrt(n)) |
| 插入删除 | O(n) | O(sqrt(n)) |
| 实现 | 简单 | 稍复杂 |

## 适用场景

- 需要频繁插入删除：块内无序，插入只需重建一个块
- 数据量大但动态变化
- 数据库索引的简化模型

## 常见陷阱

1. **块大小选择**：sqrt(n) 是理论最优，实际可能需要调整
2. **块间有序**：必须保证第 i 块最大值 < 第 i+1 块最小值
3. **重建索引**：插入/删除后需要更新索引表
