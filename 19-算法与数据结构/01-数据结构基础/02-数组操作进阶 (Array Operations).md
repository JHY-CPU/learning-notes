## 02-数组操作进阶 (Array Operations)

深入探讨数组的高级操作方法，包括 map、filter、reduce、flat 等函数式编程方法。

## 高阶函数

JavaScript 数组提供了一系列强大的高阶函数：

```javascript

// map - 映射每个元素
let doubled = [1,2,3].map(x => x * 2); // [2,4,6]

// filter - 过滤元素
let evens = [1,2,3,4].filter(x => x % 2 === 0); // [2,4]

// reduce - 归约为单个值
let sum = [1,2,3,4].reduce((acc, x) => acc + x, 0); // 10

// forEach - 遍历
[1,2,3].forEach(x => console.log(x));

// some - 任一满足
let hasLarge = [1,2,3].some(x => x > 2); // true

// every - 全部满足
let allPositive = [1,2,3].every(x => x > 0); // true

// find - 查找第一个匹配
let found = [1,2,3].find(x => x > 1); // 2

// findIndex - 查找第一个匹配的索引
let idx = [1,2,3].findIndex(x => x > 1); // 1
```

## 数组展平与拼接

```javascript

// flat - 展平嵌套数组
let flat = [1, [2, 3], [4, [5]]].flat(2); // [1,2,3,4,5]

// flatMap - map + flat 合并操作
let words = ['hello', 'world'];
let chars = words.flatMap(w => w.split('')); // ['h','e','l','l','o','w','o','r','l','d']

// concat - 拼接数组
let merged = [1,2].concat([3,4]); // [1,2,3,4]
```

## 排序与反转

```javascript

// sort - 排序（默认按字符串排序）
let nums = [3, 1, 10, 2];
nums.sort((a, b) => a - b); // [1,2,3,10] 数值升序

// reverse - 反转
nums.reverse(); // [10,3,2,1]

// slice - 切片（不修改原数组）
let sliced = [1,2,3,4,5].slice(1, 4); // [2,3,4]

// splice - 增删（修改原数组）
let arr = [1,2,3,4,5];
arr.splice(2, 1, 99); // 从索引2删除1个元素，插入99
// arr 变为 [1,2,99,4,5]
```

## 交互演示
