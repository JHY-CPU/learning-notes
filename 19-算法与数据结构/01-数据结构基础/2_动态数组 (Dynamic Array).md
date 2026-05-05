## 03-动态数组 (Dynamic Array)

动态数组是一种在运行时自动调整大小的数组。JavaScript 数组本身就是动态的，这里我们模拟实现一个动态数组来理解其原理。

## 动态数组核心概念

动态数组在底层使用静态数组存储，当容量不足时自动扩容（通常为原容量的 1.5~2 倍）。

```javascript

// 动态数组的关键属性：
// 1. data - 底层存储数组
// 2. size - 当前元素个数
// 3. capacity - 当前容量

// 扩容策略：当 size === capacity 时扩容
// 缩容策略：当 size < capacity/4 时缩容

class DynamicArray {
  constructor(capacity = 4) {
    this.data = new Array(capacity);
    this.size = 0;
    this.capacity = capacity;
  }

  // 扩容 - 创建新数组并复制数据
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

  // 末尾删除 - 均摊 O(1)
  remove() {
    if (this.size === 0) throw new Error('空数组');
    let element = this.data[--this.size];
    // 缩容
    if (this.size < this.capacity / 4 && this.capacity > 4) {
      this.resize(Math.floor(this.capacity / 2));
    }
    return element;
  }
}
```

## 时间复杂度分析

```javascript

// 访问: O(1) - 通过索引直接访问
// 修改: O(1) - 通过索引直接修改
// 末尾添加: 均摊 O(1) - 偶尔需要扩容
// 末尾删除: 均摊 O(1) - 偶尔需要缩容
// 中间插入: O(n) - 需要移动元素
// 中间删除: O(n) - 需要移动元素
// 查找: O(n) - 需要遍历
```

## 交互演示
