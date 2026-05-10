# 算法面试准备策略 (Interview Prep)

## 一、面试概览与规划

### 1.1 算法面试的本质

算法面试考察的核心能力：
- **问题分析能力：** 将模糊问题转化为具体的数据结构和算法
- **编码能力：** 在白板/编辑器中快速正确地写出代码
- **沟通能力：** 清晰表达思路，与面试官互动
- **优化能力：** 从暴力解法逐步优化到最优解

### 1.2 准备时间规划

| 阶段 | 时间 | 内容 |
|------|------|------|
| 基础夯实 | 2-3周 | 数据结构基础、常用算法模板 |
| 专题突破 | 4-6周 | 按主题刷题，每类20-30题 |
| 模拟面试 | 2-3周 | 限时练习、模拟真实面试 |
| 查漏补缺 | 1-2周 | 复习错题、薄弱环节 |

### 1.3 刷题数量建议

- **入门级：** 100-150题，覆盖高频题型
- **进阶级：** 200-300题，各类型熟练
- **竞赛级：** 500+题，包含中等和困难题

---

## 二、核心专题与优先级

### 2.1 高优先级专题（必刷）

1. **数组与字符串** — 双指针、滑动窗口、前缀和
2. **链表** — 翻转、合并、快慢指针
3. **哈希表** — 两数之和、字母异位词
4. **二叉树** — 遍历、BST操作、LCA
5. **动态规划** — 背包、LIS、LCS、区间DP
6. **二分查找** — 标准二分、二分答案
7. **BFS/DFS** — 图遍历、岛屿问题、拓扑排序

### 2.2 中优先级专题

8. **栈与队列** — 单调栈、优先队列
9. **贪心算法** — 区间调度、跳跃游戏
10. **回溯算法** — 排列组合、N皇后、数独
11. **图论** — 最短路、最小生成树、并查集
12. **排序** — 快排、归并、堆排序

### 2.3 低优先级（时间充裕时）

13. 位运算、数学、字典树、线段树
14. 高级图论（网络流、强连通分量）
15. 高级数据结构（跳表、树状数组）

---

## 三、解题方法论

### 3.1 五步解题法

```
1. 理解题意 → 确认输入输出、边界条件、数据范围
2. 举例分析 → 用具体例子手动模拟，找规律
3. 选择方案 → 暴力 → 优化 → 最优
4. 编码实现 → 先写框架，再填细节
5. 测试验证 → 小数据、边界、大数据
```

### 3.2 常见解题模式

| 模式 | 触发条件 | 典型问题 |
|------|---------|---------|
| 滑动窗口 | 连续子数组/子串满足条件 | 最长无重复子串 |
| 双指针 | 有序数组、链表中点 | 两数之和II |
| 二分搜索 | 有序、单调性、答案范围 | 矩阵搜索 |
| BFS | 最短路径、层序 | 二叉树层序 |
| DFS | 排列组合、连通性 | 岛屿数量 |
| 动态规划 | 最优值、计数、可行性 | 背包、LIS |
| 贪心 | 局部最优=全局最优 | 区间调度 |
| 回溯 | 所有方案、组合 | 全排列 |

### 3.3 复杂度分析技巧

根据数据量选择算法：
- **n ≤ 10：** 指数级 $O(2^n)$、$O(n!)$
- **n ≤ 25：** $O(2^n)$ 状态压缩
- **n ≤ 5000：** $O(n^2)$ DP
- **n ≤ 10^5：** $O(n \log n)$ 排序、分治
- **n ≤ 10^7：** $O(n)$ 线性扫描
- **n ≤ 10^9：** $O(\sqrt{n})$、$O(\log n)$

---

## 四、代码模板库

### 4.1 二分查找模板

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 找第一个满足条件的位置
def lower_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left
```

### 4.2 BFS模板

```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = {start}
    step = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if is_target(node):
                return step
            for neighbor in get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        step += 1
    return -1
```

### 4.3 DFS/回溯模板

```python
def backtrack(path, choices):
    if is_solution(path):
        result.append(path[:])
        return
    for choice in choices:
        if not is_valid(choice, path):
            continue
        path.append(choice)
        backtrack(path, remaining_choices)
        path.pop()
```

### 4.4 动态规划模板

```python
def dp_template(nums):
    n = len(nums)
    dp = [0] * (n + 1)  # 或二维数组
    dp[0] = base_value

    for i in range(1, n + 1):
        for j in range(i):  # 根据问题调整
            dp[i] = optimize(dp[i], dp[j] + cost(j, i))

    return dp[n]
```

---

## 五、面试当天技巧

### 5.1 沟通框架

```
1. 确认理解："我确认一下题意，输入是...，输出是...，对吗？"
2. 暴力思路："最直接的思路是...，时间复杂度是...，但我们可以优化。"
3. 优化过程："如果我们用...，可以把复杂度降到...。"
4. 编码时："我先写主框架，再处理边界。"
5. 测试："让我用这个例子走一遍..."
```

### 5.2 卡住时的应对

1. 从小例子开始手动模拟
2. 考虑是否能用已知模板解决
3. 换一种数据结构试试
4. 主动和面试官讨论思路
5. 先写暴力解法再优化

### 5.3 常见错误与避免

- **边界条件遗漏：** 空数组、单元素、全相同
- **整数溢出：** 用 `long` 或 `mid = left + (right-left)//2`
- **死循环：** 二分时注意 `left = mid+1` 和 `right = mid-1`
- **内存泄漏：** C++中 `new` 要对应 `delete`

---

## 六、刷题平台与资源

### 6.1 推荐平台

- **LeetCode：** 面试刷题首选，题目分类清晰
- **牛客网：** 国内面试题库丰富
- **Codeforces：** 竞赛训练，提升思维
- **AtCoder：** 题目质量高，适合进阶

### 6.2 推荐题单

1. **LeetCode Hot 100：** 面试高频题
2. **LeetCode 剑指Offer：** 国内面试必刷
3. **Blind 75：** 精选75题覆盖主要类型
4. **Grind 75：** Blind 75的升级版，可自定义时间

### 6.3 学习方法

- **每日一题：** 保持手感
- **限时练习：** 中等题20分钟，困难题40分钟
- **复习错题：** 3天、7天、14天间隔复习
- **总结模板：** 每类题型总结通用模板
