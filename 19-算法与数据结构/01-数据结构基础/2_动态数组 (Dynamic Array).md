# 03-动态数组 (Dynamic Array)

动态数组是一种在运行时自动调整大小的数组。JavaScript 数组本身就是动态的，这里我们模拟实现一个动态数组来理解其原理。

## 动态数组核心概念

动态数组在底层使用静态数组存储，当容量不足时自动扩容（通常为原容量的 1.5~2 倍）。

```javascript
// 动态数组的关键属性：
// 1. data - 底层存储数组
// 2. size - 当前元素个数
// 3. capacity - 当前容量

class DynamicArray {
  constructor(capacity = 4) {
    this.data = new Array(capacity);
    this.size = 0;
    this.capacity = capacity;
  }

  // 扩容 - 创建新数组并复制数据 O(n)
  resize(newCapacity) {
    let newData = new Array(newCapacity);
    for (let i = 0; i < this.size; i++) {
      newData[i] = this.data[i];
    }
    this.data = newData;
    this.capacity = newCapacity;
  }

  // 末尾添加 - 均摊 O(1)
  add(element) {
    if (this.size === this.capacity) {
      this.resize(this.capacity * 2);
    }
    this.data[this.size++] = element;
  }

  // 按索引访问 - O(1)
  get(index) {
    if (index < 0 || index >= this.size) throw new Error('索引越界');
    return this.data[index];
  }

  // 按索引修改 - O(1)
  set(index, element) {
    if (index < 0 || index >= this.size) throw new Error('索引越界');
    this.data[index] = element;
  }

  // 末尾删除 - 均摊 O(1)
  remove() {
    if (this.size === 0) throw new Error('空数组');
    let element = this.data[--this.size];
    if (this.size < this.capacity / 4 && this.capacity > 4) {
      this.resize(Math.floor(this.capacity / 2));
    }
    return element;
  }

  // 中间插入 - O(n)
  insert(index, element) {
    if (index < 0 || index > this.size) throw new Error('索引越界');
    if (this.size === this.capacity) this.resize(this.capacity * 2);
    for (let i = this.size; i > index; i--) {
      this.data[i] = this.data[i - 1];
    }
    this.data[index] = element;
    this.size++;
  }
}
```

## C++ 实现

```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

template<typename T>
class DynamicArray {
    T* data;
    int sz;
    int cap;

    void resize(int newCap) {
        T* newData = new T[newCap];
        for (int i = 0; i < sz; i++) {
            newData[i] = data[i];
        }
        delete[] data;
        data = newData;
        cap = newCap;
    }

public:
    DynamicArray(int capacity = 4) : sz(0), cap(capacity) {
        data = new T[cap];
    }

    ~DynamicArray() { delete[] data; }

    void push_back(const T& val) {
        if (sz == cap) resize(cap * 2);
        data[sz++] = val;
    }

    void pop_back() {
        if (sz == 0) throw runtime_error("empty");
        sz--;
        if (sz < cap / 4 && cap > 4) resize(cap / 2);
    }

    T& operator[](int i) { return data[i]; }
    int size() const { return sz; }
    int capacity() const { return cap; }
};
```

## 均摊分析

扩容操作虽然单次是 O(n)，但均摊到每次添加操作是 O(1)：

- 假设初始容量为 1，执行 n 次 add 操作
- 扩容发生在第 1, 2, 4, 8, ... 次添加时
- 总复制次数 = 1 + 2 + 4 + 8 + ... + n/2 = 2n - 1
- 均摊每次操作 = (n + 2n - 1) / n ≈ O(1)

## 扩容策略对比

| 策略 | 倍数 | 优点 | 缺点 |
|------|------|------|------|
| 2倍扩容 | ×2 | 均摊 O(1)，简单 | 可能浪费较多内存 |
| 1.5倍扩容 | ×1.5 | 内存复用更好 | 均摊常数略大 |
| 固定增量 | +N | 内存紧凑 | 均摊 O(√n) |

Java ArrayList 使用 1.5 倍，C++ vector 使用 2 倍。

## 何时使用动态数组 vs 链表

| 场景 | 动态数组 | 链表 |
|------|---------|------|
| 随机访问 | O(1) 优胜 | O(n) |
| 尾部插入 | O(1) 均摊 | O(1) |
| 中间插入 | O(n) | O(1)（已找到位置） |
| 内存开销 | 较小 | 每个节点有指针开销 |
| 缓存友好 | 是 | 否 |

大多数场景下动态数组更优，除非频繁在中间插入/删除且不需要随机访问。
