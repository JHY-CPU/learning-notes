# 40-算法总结 (Algorithms Summary)

将本章学习的算法基础核心思想进行系统总结，帮助形成完整的知识框架。

## 核心思想对比

| 算法 | 核心思想 | 适用条件 | 典型复杂度 |
|------|---------|----------|-----------|
| 贪心 | 局部最优->全局最优 | 贪心选择性质 + 最优子结构 | O(n)~O(n log n) |
| 分治 | 分解->解决->合并 | 子问题独立 | O(n log n) |
| 二分 | 缩半搜索 | 有序性/单调性 | O(log n) |
| DP | 状态定义 + 转移方程 | 最优子结构 + 重叠子问题 | O(n²)~O(n) |
| 回溯 | 决策树遍历 + 剪枝 | 需要求所有解 | 指数级 |
| 随机化 | 随机选择简化问题 | 期望高效 | 视情况而定 |
| 近似 | 有理论保证的近似 | NP-hard 问题 | 多项式 |
| 在线 | 无未来信息的实时决策 | 实时系统 | 竞争比分析 |

## 选算法的思考路径

```
1. 数据是否有序？-> 二分查找
2. 能否贪心？-> 检查贪心选择性质
3. 能否分治？-> 子问题是否独立且可合并
4. 有无重叠子问题？-> 动态规划
5. 需要所有解？-> 回溯 + 剪枝
6. 问题是 NP-hard？-> 近似算法或随机化
7. 输入逐个到达？-> 在线算法
```

## JavaScript 实现：算法选择框架

```javascript
// 算法选择指南
const algorithmGuide = {
  '贪心': {
    condition: '贪心选择性质 + 最优子结构',
    examples: ['区间调度', '分发饼干', '跳跃游戏', '加油站'],
    complexity: 'O(n) ~ O(n log n)',
    verify: '交换论证法'
  },
  '分治': {
    condition: '子问题独立，可递归分解',
    examples: ['归并排序', '快速排序', '最近点对', '大整数乘法'],
    complexity: 'O(n log n)',
    verify: '数学归纳法'
  },
  '二分': {
    condition: '有序性或答案单调性',
    examples: ['查找元素', '二分答案', '旋转数组', '搜索范围'],
    complexity: 'O(log n)',
    verify: '区间缩半，必收敛'
  },
  '动态规划': {
    condition: '最优子结构 + 重叠子问题',
    examples: ['背包问题', '最长子序列', '编辑距离', '区间 DP'],
    complexity: 'O(n²) ~ O(n)',
    verify: '最优子结构证明'
  },
  '回溯': {
    condition: '需要枚举所有解或最优解',
    examples: ['全排列', '组合', 'N 皇后', '数独'],
    complexity: '指数级',
    verify: '剪枝 + 对拍'
  }
};

// 打印算法选择指南
for (const [algo, info] of Object.entries(algorithmGuide)) {
  console.log(`\n== ${algo} ==`);
  console.log(`条件: ${info.condition}`);
  console.log(`典型问题: ${info.examples.join(', ')}`);
  console.log(`复杂度: ${info.complexity}`);
  console.log(`验证方法: ${info.verify}`);
}
```

## C++ 实现：常用算法模板

```cpp
#include <vector>
#include <algorithm>
using namespace std;

// 贪心模板
int greedyTemplate(vector<int>& arr) {
    sort(arr.begin(), arr.end(), /* 自定义比较 */);
    int result = 0;
    for (auto& x : arr) {
        // 贪心选择
        result += /* 贡献 */;
    }
    return result;
}

// 分治模板
int divideConquer(vector<int>& arr, int l, int r) {
    if (l >= r) return /* 基础情况 */;
    int mid = l + (r - l) / 2;
    int left = divideConquer(arr, l, mid);
    int right = divideConquer(arr, mid + 1, r);
    return /* 合并 left 和 right */;
}

// DP 模板
int dpTemplate(vector<int>& arr) {
    int n = arr.size();
    vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        dp[i] = /* 转移方程 */;
    }
    return dp[n];
}

// 回溯模板
void backtrack(vector<int>& state, vector<int>& choices, vector<vector<int>>& result) {
    if (/* 终止条件 */) {
        result.push_back(state);
        return;
    }
    for (int i = 0; i < choices.size(); i++) {
        if (/* 剪枝 */) continue;
        state.push_back(choices[i]);
        backtrack(state, choices, result);
        state.pop_back();
    }
}
```

## 算法关系图

```
        排序 (O(n log n))
       /    \
   贪心      分治
    |         |
    v         v
  区间问题   归并/快排
    |
    v
  二分查找 -> 二分答案
    |
    v
  动态规划 (重叠子问题)
    |
    v
  回溯 (穷举 + 剪枝)
    |
    v
  NP-hard -> 近似/随机化
```

## 复杂度速查

| 复杂度 | n=1e5 | n=1e6 | 代表算法 |
|--------|-------|-------|---------|
| O(1) | 快 | 快 | 哈希查找 |
| O(log n) | 快 | 快 | 二分查找 |
| O(n) | 100ms | 1s | 线性扫描 |
| O(n log n) | 1.7s | 20s | 排序 |
| O(n²) | 100s | 不可行 | 暴力枚举 |
| O(2^n) | 不可行 | 不可行 | 暴力子集 |

## 常见陷阱

1. **贪心不证**：不是所有局部最优都能导出全局最优，必须验证
2. **DP 状态不足**：状态定义不完整会导致错误答案
3. **二分死循环**：l = mid + 1 / r = mid - 1 写错导致死循环
4. **回溯超时**：忘记剪枝导致指数级时间爆炸
5. **复杂度误判**：嵌套循环不一定是 O(n²)，要看内部指针移动方式

## 实际应用

刷题时按算法分类练习，每类至少做 10 题。遇到新题先尝试分类（属于哪种模式），再套用模板。随着经验积累，分类判断会越来越快。建议建立自己的模板库，面试时能快速套用。
