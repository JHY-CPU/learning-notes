# 面试高频专题 (Interview Hot)

## 一、概念定义与原理

本专题汇总 LeetCode Hot 100 中最高频的算法题目，按照面试中出现频率排序。

### 1.1 面试考察重点

1. **数据结构：** 数组、链表、树、图、哈希
2. **算法思想：** 双指针、滑动窗口、BFS/DFS、动态规划、贪心
3. **代码能力：** 边界处理、代码简洁性
4. **沟通能力：** 分析复杂度、讨论优化

---

## 二、高频题型 Top 20

### 2.1 数组与字符串

| 题号 | 题目 | 核心技巧 |
|------|------|---------|
| 1 | 两数之和 | 哈希表 |
| 3 | 无重复字符的最长子串 | 滑动窗口 |
| 11 | 盛最多水的容器 | 对撞指针 |
| 15 | 三数之和 | 排序+对撞指针 |
| 53 | 最大子数组和 | Kadane/Dynamic Programming |

### 2.2 链表

| 题号 | 题目 | 核心技巧 |
|------|------|---------|
| 206 | 反转链表 | 三指针法 |
| 21 | 合并两个有序链表 | 归并 |
| 141 | 环形链表 | 快慢指针 |
| 19 | 删除链表的倒数第N个节点 | 双指针 |

### 2.3 树

| 题号 | 题目 | 核心技巧 |
|------|------|---------|
| 102 | 二叉树的层序遍历 | BFS |
| 236 | 二叉树的最近公共祖先 | 递归 |
| 104 | 二叉树的最大深度 | 递归/BFS |
| 543 | 二叉树的直径 | DFS |

### 2.4 动态规划

| 题号 | 题目 | 核心技巧 |
|------|------|---------|
| 70 | 爬楼梯 | 线性DP |
| 300 | 最长递增子序列 | DP+二分 |
| 322 | 零钱兑换 | 背包DP |
| 1143 | 最长公共子序列 | 二维DP |

### 2.5 图与搜索

| 题号 | 题目 | 核心技巧 |
|------|------|---------|
| 200 | 岛屿数量 | BFS/DFS |
| 207 | 课程表 | 拓扑排序 |
| 994 | 腐烂的橘子 | 多源BFS |

---

## 三、代码实现（精选）

### 3.1 三数之和 - C++

```cpp
vector<vector<int>> three_sum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> result;
    for (int i = 0; i < nums.size(); i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue;
        int l = i+1, r = nums.size()-1;
        while (l < r) {
            int sum = nums[i] + nums[l] + nums[r];
            if (sum == 0) {
                result.push_back({nums[i], nums[l], nums[r]});
                while (l < r && nums[l] == nums[l+1]) l++;
                while (l < r && nums[r] == nums[r-1]) r--;
                l++; r--;
            } else if (sum < 0) l++;
            else r--;
        }
    }
    return result;
}
```

### 3.2 岛屿数量 - C++

```cpp
int num_islands(vector<vector<char>>& grid) {
    int m = grid.size(), n = grid[0].size(), count = 0;
    int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '1') {
                count++;
                queue<pair<int,int>> q; q.push({i,j}); grid[i][j] = '0';
                while (!q.empty()) {
                    auto [x,y] = q.front(); q.pop();
                    for (int d = 0; d < 4; d++) {
                        int nx = x+dx[d], ny = y+dy[d];
                        if (nx>=0 && nx<m && ny>=0 && ny<n && grid[nx][ny]=='1') {
                            grid[nx][ny] = '0'; q.push({nx,ny});
                        }
                    }
                }
            }
        }
    }
    return count;
}
```

### 3.3 Python 实现

```python
def three_sum(nums):
    nums.sort(); result = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                result.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l+1]: l += 1
                while l < r and nums[r] == nums[r-1]: r -= 1
                l += 1; r -= 1
            elif s < 0: l += 1
            else: r -= 1
    return result

def max_subarray(nums):
    result = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        result = max(result, cur)
    return result

print(three_sum([-1,0,1,2,-1,-4]))  # [[-1,-1,2],[-1,0,1]]
print(max_subarray([-2,1,-3,4,-1,2,1,-5,4]))  # 6
```

### 3.4 LRU Cache（面试高频）

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

---

## 四、面试策略

### 4.1 解题模板

1. **理解题意：** 复述题目，确认输入输出
2. **举例验证：** 用示例手动模拟
3. **暴力方法：** 先说暴力思路
4. **优化思路：** 分析瓶颈，提出优化
5. **编码实现：** 边写边解释
6. **测试验证：** 用示例测试，考虑边界

### 4.2 常见陷阱

- 空输入
- 单元素数组
- 整数溢出
- 链表/树的空指针
- 重复元素处理

---

## 五、高频知识点速查

| 类别 | 必刷题 |
|------|--------|
| 二分查找 | 33, 34, 69 |
| 双指针 | 1, 11, 15, 42 |
| 滑动窗口 | 3, 76, 438 |
| BFS/DFS | 200, 207, 994 |
| 动态规划 | 53, 70, 300, 322 |
| 树 | 94, 102, 236, 543 |
| 链表 | 21, 141, 206 |
