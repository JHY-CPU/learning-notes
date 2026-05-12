# 02-数组操作进阶 (Array Operations)

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

// 展开运算符拼接
let merged2 = [...[1,2], ...[3,4]]; // [1,2,3,4]
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

## C++ 等价操作

```cpp
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

int main() {
    vector<int> v = {3, 1, 4, 1, 5, 9};

    // 排序
    sort(v.begin(), v.end());

    // 反转
    reverse(v.begin(), v.end());

    // 查找
    auto it = find(v.begin(), v.end(), 4);

    // 累加
    int sum = accumulate(v.begin(), v.end(), 0);

    // 过滤（复制满足条件的元素）
    vector<int> result;
    copy_if(v.begin(), v.end(), back_inserter(result),
            [](int x) { return x % 2 == 0; });

    // 变换
    vector<int> doubled(v.size());
    transform(v.begin(), v.end(), doubled.begin(),
              [](int x) { return x * 2; });

    // 删除元素
    v.erase(remove(v.begin(), v.end(), 1), v.end()); // 删除所有1

    return 0;
}
```

## reduce 高级用法

```javascript
// 分组
let people = [
  {name: 'Alice', age: 25},
  {name: 'Bob', age: 30},
  {name: 'Carol', age: 25}
];
let grouped = people.reduce((acc, p) => {
  (acc[p.age] = acc[p.age] || []).push(p);
  return acc;
}, {});

// 计数
let fruits = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple'];
let counts = fruits.reduce((acc, f) => {
  acc[f] = (acc[f] || 0) + 1;
  return acc;
}, {});

// 管道
let result = [1,2,3,4,5]
  .filter(x => x % 2 === 0)
  .map(x => x * x)
  .reduce((a, b) => a + b, 0); // 4 + 16 = 20
```

## 性能注意事项

1. **链式调用**：每次调用 map/filter 都会创建新数组，大数据量时考虑用 reduce 一次完成
2. **forEach vs for**：传统 for 循环在极端性能场景下更快
3. **惰性求值**：JavaScript 没有内置惰性数组，大数据可用 generator 模拟
4. **原地操作**：sort、reverse、splice 会修改原数组，需注意副作用

## 常见陷阱

1. **sort 的数值排序**：`[10, 2, 1].sort()` 不等于 `[1, 2, 10]`，需传入比较函数
2. **map 的稀疏数组**：`[1, , 3].map(x => x * 2)` 跳过空洞，结果为 `[2, empty, 6]`
3. **reduce 无初始值**：空数组调用 reduce 无初始值会报错
4. **splice 返回值**：返回被删除的元素数组，不是修改后的数组
