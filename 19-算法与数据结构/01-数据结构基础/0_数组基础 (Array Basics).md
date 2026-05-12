# 01-数组基础 (Array Basics)

数组是最基础的数据结构，用于在连续内存中存储相同类型的元素。JavaScript 中的数组是动态的，可以存储任意类型。

## 数组定义与创建

JavaScript 中创建数组的多种方式：

```javascript
// 字面量方式
let arr1 = [1, 2, 3, 4, 5];

// 构造函数方式
let arr2 = new Array(5); // 长度为5的空数组

// Array.of 方式
let arr3 = Array.of(1, 2, 3);

// Array.from 方式
let arr4 = Array.from('hello'); // ['h','e','l','l','o']
```

## 基本操作

数组的增删改查操作：

```javascript
let nums = [10, 20, 30, 40, 50];

// 访问元素 O(1)
console.log(nums[2]); // 30

// 修改元素 O(1)
nums[2] = 35;

// 末尾添加 O(1)
nums.push(60);

// 末尾删除 O(1)
nums.pop();

// 开头添加 O(n)
nums.unshift(5);

// 开头删除 O(n)
nums.shift();

// 获取长度
console.log(nums.length);
```

## C++ 中的数组

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    // 静态数组
    int arr[5] = {10, 20, 30, 40, 50};

    // 访问 O(1)
    cout << arr[2] << endl; // 30

    // 动态数组 vector
    vector<int> v = {10, 20, 30};
    v.push_back(40);   // 末尾添加 O(1) 均摊
    v.pop_back();       // 末尾删除 O(1)
    v.insert(v.begin() + 1, 15); // 中间插入 O(n)
    v.erase(v.begin());           // 开头删除 O(n)

    cout << v.size() << endl;
    cout << v[0] << endl;

    return 0;
}
```

## 时间复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 按索引访问 | O(1) | 连续内存直接偏移 |
| 按索引修改 | O(1) | 同上 |
| 末尾添加/删除 | O(1) 均摊 | 偶尔需要扩容 |
| 开头添加/删除 | O(n) | 需要移动所有元素 |
| 中间插入/删除 | O(n) | 需要移动后续元素 |
| 线性查找 | O(n) | 最坏遍历全部 |
| 二分查找 | O(log n) | 需要有序数组 |

## 内存布局与缓存友好性

数组元素在内存中连续存储，这带来两个重要优势：

1. **缓存友好**：CPU 缓存可以预取相邻元素，访问效率极高
2. **随机访问**：通过 `base_address + index * element_size` 直接定位

这就是为什么在性能敏感场景中，数组往往优于链表，即使理论上某些操作链表更优。

## 常见陷阱

1. **数组长度与索引**：`arr.length` 是长度，最大索引是 `length - 1`
2. **稀疏数组**：JavaScript 中 `[1, , 3]` 和 `[1, undefined, 3]` 行为不同
3. **引用赋值**：`let b = a` 只是引用，修改 b 会影响 a
4. **sort 的默认行为**：`[10, 2, 1].sort()` 结果是 `[1, 10, 2]`（按字符串排序）

## 何时使用数组

- 需要频繁按索引访问元素
- 数据量已知或可预估
- 需要缓存友好的数据结构
- 实现栈、队列等基础数据结构
