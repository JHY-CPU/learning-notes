## 01-数组基础 (Array Basics)

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

// 创建数组
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

## 交互演示
