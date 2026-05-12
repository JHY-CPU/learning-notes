# 30-算法可视化与调试 (Visualization & Debugging)

算法调试比普通程序更难，因为逻辑更抽象。使用可视化和系统化调试方法可以快速定位错误。

## 调试方法

| 方法 | 适用场景 | 效率 |
|------|---------|------|
| 打印中间状态 | 循环/递归逻辑错误 | 高 |
| 小规模测试 | 验证基本逻辑 | 高 |
| 图形化展示 | 树/图结构问题 | 中 |
| 对拍验证 | 优化算法验证 | 高 |
| 边界测试 | 边界条件遗漏 | 高 |
| 断点调试 | 复杂状态跟踪 | 中 |

## JavaScript 实现

```javascript
// 排序过程可视化（打印每次交换）
function debugSort(arr) {
  const steps = [[...arr]];
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr.length - 1 - i; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        steps.push([...arr]);
      }
    }
  }
  return steps;
}

// 递归调用栈追踪
function debugFib(n, depth = 0) {
  const indent = '  '.repeat(depth);
  console.log(`${indent}fib(${n}) 开始`);
  if (n <= 1) {
    console.log(`${indent}  返回 ${n}`);
    return n;
  }
  const result = debugFib(n - 1, depth + 1) + debugFib(n - 2, depth + 1);
  console.log(`${indent}  返回 ${result}`);
  return result;
}

// 二分查找调试
function debugBinarySearch(arr, target) {
  let l = 0, r = arr.length - 1;
  let step = 0;
  while (l <= r) {
    const mid = Math.floor((l + r) / 2);
    console.log(`步骤${++step}: l=${l}, r=${r}, mid=${mid}, arr[mid]=${arr[mid]}`);
    if (arr[mid] === target) {
      console.log(`找到目标! 索引=${mid}`);
      return mid;
    }
    if (arr[mid] < target) l = mid + 1;
    else r = mid - 1;
  }
  console.log(`未找到`);
  return -1;
}

// 对拍函数：比较暴力解法与优化解法
function stressTest(optimizedFn, bruteForceFn, generateInput, iterations = 1000) {
  for (let i = 0; i < iterations; i++) {
    const input = generateInput();
    const expected = bruteForceFn(input);
    const actual = optimizedFn(input);
    if (JSON.stringify(expected) !== JSON.stringify(actual)) {
      console.error(`测试失败! 输入: ${JSON.stringify(input)}`);
      console.error(`期望: ${expected}, 实际: ${actual}`);
      return false;
    }
  }
  console.log(`全部 ${iterations} 组测试通过`);
  return true;
}

// 测试
console.log(debugSort([4, 2, 5, 1, 3]));
debugBinarySearch([1, 3, 5, 7, 9, 11], 7);
```

## C++ 实现

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

// 对拍模板
int bruteForce(vector<int>& arr) {
    // 暴力解法
    int maxSum = INT_MIN;
    for (int i = 0; i < arr.size(); i++) {
        int sum = 0;
        for (int j = i; j < arr.size(); j++) {
            sum += arr[j];
            maxSum = max(maxSum, sum);
        }
    }
    return maxSum;
}

int optimized(vector<int>& arr) {
    // Kadane 算法
    int maxSum = arr[0], curSum = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        curSum = max(arr[i], curSum + arr[i]);
        maxSum = max(maxSum, curSum);
    }
    return maxSum;
}

// 随机数据生成 + 对拍
void stressTest(int iterations = 10000) {
    srand(42);
    for (int t = 0; t < iterations; t++) {
        int n = rand() % 20 + 1;
        vector<int> arr(n);
        for (int i = 0; i < n; i++) arr[i] = rand() % 200 - 100;

        int expected = bruteForce(arr);
        int actual = optimized(arr);

        if (expected != actual) {
            cout << "测试失败! 输入: ";
            for (int x : arr) cout << x << " ";
            cout << "\n期望: " << expected << ", 实际: " << actual << endl;
            return;
        }
    }
    cout << "全部 " << iterations << " 组测试通过" << endl;
}
```

## 常见错误来源

| 错误类型 | 表现 | 调试方法 |
|----------|------|---------|
| Off-by-one | 差 1 的错误 | 打印边界值 |
| 整数溢出 | 大数据结果异常 | 换用 long long |
| 递归终止 | 栈溢出 | 打印调用栈 |
| 数组越界 | 崩溃/随机值 | 检查索引范围 |
| 浮点精度 | 比较失败 | 用 eps |

## 调试流程

1. 用最小数据（n=1,2,3）手工验证
2. 打印关键变量的中间值
3. 与暴力解法对拍
4. 测试边界：空、单元素、最大值
5. 随机大数据压力测试

## 常见陷阱

1. **过度依赖打印**：打印太多反而难以定位
2. **忽略随机性**：随机化算法需要固定种子才能复现
3. **对拍太慢**：暴力解法也要保证正确
4. **不测边界**：空输入和单元素是最常见的 bug 来源

## 实际应用

写完算法后先用小数据手工跑一遍，再用大数据对拍。竞赛中最高效的调试方式是对拍：随机生成数据，暴力和优化版本同时运行对比结果。线上 debug 时，二分定位法（逐步缩小出错范围）最有效。
