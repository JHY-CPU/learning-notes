# 33-算法复杂度实验 (Complexity Experiment)

通过实际运行观察不同算法的性能差异，验证理论复杂度分析的正确性。

## 实验方法

1. 固定其他变量，只改变输入规模 n
2. 对每个 n 多次运行取平均值
3. 绘制 n-time 图，观察增长趋势
4. 与理论复杂度对比验证

## JavaScript 实现

```javascript
// 计时工具
function measureTime(fn, ...args) {
  const iterations = 5;
  let total = 0;
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn(...args);
    total += performance.now() - start;
  }
  return total / iterations;
}

// O(n) vs O(n²) vs O(n log n) 对比
function linear(n) {
  let s = 0;
  for (let i = 0; i < n; i++) s += i;
  return s;
}

function quadratic(n) {
  let s = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) s += j;
  return s;
}

function nlogn(n) {
  const arr = Array.from({ length: n }, () => Math.random());
  arr.sort((a, b) => a - b);  // 内置排序 O(n log n)
  return arr[0];
}

// 实验对比
function runExperiment() {
  const sizes = [1000, 5000, 10000, 50000, 100000];
  console.log('n\t\tO(n)\t\tO(n log n)\tO(n²)');
  for (const n of sizes) {
    const t1 = measureTime(linear, n);
    const t2 = measureTime(nlogn, n);
    const t3 = n <= 10000 ? measureTime(quadratic, n) : 'N/A'; // O(n²) 太慢跳过
    console.log(`${n}\t\t${t1.toFixed(2)}ms\t\t${t2.toFixed(2)}ms\t\t${typeof t3 === 'number' ? t3.toFixed(2) + 'ms' : t3}`);
  }
}

// 验证增长趋势
function verifyComplexity() {
  console.log('\n--- 增长趋势验证 ---');
  const ns = [1000, 2000, 4000, 8000];
  let prevTime = 0;
  for (const n of ns) {
    const t = measureTime(linear, n);
    if (prevTime > 0) {
      const ratio = t / prevTime;
      console.log(`n: ${n}, 时间: ${t.toFixed(2)}ms, 倍率: ${ratio.toFixed(2)}x`);
    } else {
      console.log(`n: ${n}, 时间: ${t.toFixed(2)}ms`);
    }
    prevTime = t;
  }
  // O(n) 理论倍率: n 翻倍 → 时间翻倍
}

runExperiment();
verifyComplexity();
```

## C++ 实现

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
using namespace std;
using namespace std::chrono;

// 计时器
class Timer {
    steady_clock::time_point start;
public:
    Timer() : start(steady_clock::now()) {}
    double elapsed() {
        return duration_cast<microseconds>(steady_clock::now() - start).count() / 1000.0;
    }
};

long long linear(int n) {
    long long s = 0;
    for (int i = 0; i < n; i++) s += i;
    return s;
}

long long quadratic(int n) {
    long long s = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) s += j;
    return s;
}

void nlogn_test(int n) {
    vector<int> arr(n);
    for (int i = 0; i < n; i++) arr[i] = rand();
    sort(arr.begin(), arr.end());
}

int main() {
    int sizes[] = {1000, 5000, 10000, 50000, 100000};

    cout << "n\t\tO(n)\t\tO(n log n)\tO(n^2)" << endl;
    for (int n : sizes) {
        Timer t1; linear(n); double ms1 = t1.elapsed();
        Timer t2; nlogn_test(n); double ms2 = t2.elapsed();
        double ms3 = (n <= 10000) ? (Timer t3, quadratic(n), t3.elapsed()) : -1;

        cout << n << "\t\t" << ms1 << "ms\t\t" << ms2 << "ms\t\t";
        if (ms3 >= 0) cout << ms3 << "ms"; else cout << "N/A";
        cout << endl;
    }

    // 验证增长趋势
    cout << "\n--- O(n) 增长验证 ---" << endl;
    double prev = 0;
    for (int n : {10000, 20000, 40000, 80000}) {
        Timer t; linear(n); double ms = t.elapsed();
        if (prev > 0) cout << "n=" << n << " 时间=" << ms << "ms 倍率=" << ms / prev << "x" << endl;
        else cout << "n=" << n << " 时间=" << ms << "ms" << endl;
        prev = ms;
    }
    return 0;
}
```

## 预期结果

| n | O(n) | O(n log n) | O(n²) |
|---|------|-----------|-------|
| 1000 | ~0.01ms | ~0.1ms | ~1ms |
| 10000 | ~0.1ms | ~1ms | ~100ms |
| 100000 | ~1ms | ~10ms | ~10s |
| 1000000 | ~10ms | ~100ms | 不可行 |

## 实验要点

1. **多次取平均**：减少系统调度干扰
2. **预热**：JIT 编译/缓存会影响首次运行
3. **关注增长率**：n 翻倍时时间的变化倍率
4. **排除常数**：不同机器绝对值不同，但倍率一致

## 常见陷阱

1. **测量误差**：短时间测量误差大，需要足够大的 n
2. **编译优化**：C++ -O2 可能消除"无用"计算
3. **GC 干扰**：JavaScript 垃圾回收会引入波动
4. **只看绝对值**：应关注 n 变化时时间的变化趋势

## 实际应用

复杂度实验是验证算法分析是否正确的可靠方法。在面试中能通过实验验证自己的分析，在工程中能帮助选择最优实现方案。
