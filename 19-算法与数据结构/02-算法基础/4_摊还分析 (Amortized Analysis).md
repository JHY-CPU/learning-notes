# 05-摊还分析 (Amortized Analysis)

摊还分析将偶尔的高成本操作均摊到所有操作中，得到平均成本。区别于平均情况分析，摊还分析不依赖概率。

## 三种方法

### 1. 聚合分析

```javascript
// n 次 push 操作的总成本
// 扩容发生在第 1, 2, 4, 8, ..., n 次
// 总复制次数 = 1 + 2 + 4 + ... + n = 2n - 1
// 摊还成本 = (n + 2n - 1) / n ≈ 3 = O(1)
```

### 2. 记账法

```javascript
// 每次 push 收费 3 个单位
// 正常插入花 1 个，存 2 个作为"信用"
// 扩容时用信用支付复制成本
// 信用足够覆盖所有复制操作
```

### 3. 势能法

```javascript
// 定义势能函数 Φ
// 摊还成本 = 实际成本 + ΔΦ
// 对于动态数组：Φ = 2 * size - capacity
// 证明 Φ >= 0 且 Φ(最终) >= Φ(初始)
```

## 动态数组摊还分析

```javascript
class DynamicArray {
  constructor() {
    this.data = new Array(1);
    this.size = 0;
    this.capacity = 1;
  }

  push(x) {
    if (this.size === this.capacity) {
      // 扩容：复制所有元素 O(n)
      const newData = new Array(this.capacity * 2);
      for (let i = 0; i < this.size; i++) newData[i] = this.data[i];
      this.data = newData;
      this.capacity *= 2;
    }
    this.data[this.size++] = x; // O(1)
  }
}
// n 次 push 摊还 O(1)
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

// vector 的 push_back 就是摊还 O(1)
// capacity 不足时翻倍复制，但均摊后每次操作 O(1)
void demo() {
    vector<int> v;
    // 预留空间减少扩容次数
    v.reserve(1000); // O(n) 一次性分配

    for (int i = 0; i < 1000; i++) {
        v.push_back(i); // 摊还 O(1)
    }
}
```

## 二进制计数器

```javascript
// 从 0 计到 n，每次 increment
// 第 i 位翻转次数 = n / 2^i
// 总翻转次数 = n + n/2 + n/4 + ... ≈ 2n
// 每次 increment 摊还 O(1)

class BinaryCounter {
  constructor() { this.bits = []; this.n = 0; }

  increment() {
    let i = 0;
    while (this.bits[i] === 1) {
      this.bits[i] = 0; // 翻转
      i++;
    }
    this.bits[i] = 1; // 翻转
    this.n++;
  }
}
```

## 栈的多弹出

```javascript
// n 次 push/popMulti 操作
// popMulti(k) 弹出 k 个元素
// 但弹出总数不超过 push 总数
// 所以总成本 O(n)，每次操作摊还 O(1)
```

## 摊还 vs 平均

| 特性 | 摊还分析 | 平均情况分析 |
|------|---------|------------|
| 依赖概率 | 不依赖 | 依赖输入分布 |
| 保证类型 | 确定性 | 概率性 |
| 适用 | 数据结构操作序列 | 单次操作分析 |
| 结论 | 每次操作平均代价 | 期望代价 |

## 应用场景

- 动态数组 push/pop
- HashMap 扩容（rehash）
- 字符串拼接（某些语言）
- 斐波那契堆的 decrease-key
- Splay 树的访问操作

## 常见陷阱

1. **混淆平均情况**：摊还是对操作序列的分析，不是对输入的分析
2. **忽略扩容**：动态数组扩容是 O(n)，但摊还是 O(1)
3. **势能函数**：需要正确设计并证明非负性
