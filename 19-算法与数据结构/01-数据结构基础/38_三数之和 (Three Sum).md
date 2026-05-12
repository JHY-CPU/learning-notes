# 39-三数之和 (Three Sum)

三数之和是经典面试题，使用排序+双指针实现 O(n²) 解法，关键是去重处理。

## 排序 + 双指针解法

```javascript
// 找出所有和为 0 的三元组（不重复）
function threeSum(nums) {
  nums.sort((a, b) => a - b);
  const res = [];

  for (let i = 0; i < nums.length - 2; i++) {
    // 剪枝：最小值大于0则无解
    if (nums[i] > 0) break;
    // 去重：跳过重复的第一个数
    if (i > 0 && nums[i] === nums[i - 1]) continue;

    let l = i + 1, r = nums.length - 1;
    while (l < r) {
      const sum = nums[i] + nums[l] + nums[r];
      if (sum === 0) {
        res.push([nums[i], nums[l], nums[r]]);
        // 去重：跳过重复的第二个和第三个数
        while (l < r && nums[l] === nums[l + 1]) l++;
        while (l < r && nums[r] === nums[r - 1]) r--;
        l++; r--;
      } else if (sum < 0) {
        l++;
      } else {
        r--;
      }
    }
  }
  return res;
}

console.log(threeSum([-1, 0, 1, 2, -1, -4]));
// [[-1, -1, 2], [-1, 0, 1]]
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    for (int i = 0; i < (int)nums.size() - 2; i++) {
        if (nums[i] > 0) break;
        if (i > 0 && nums[i] == nums[i-1]) continue;
        int l = i + 1, r = nums.size() - 1;
        while (l < r) {
            int sum = nums[i] + nums[l] + nums[r];
            if (sum == 0) {
                res.push_back({nums[i], nums[l], nums[r]});
                while (l < r && nums[l] == nums[l+1]) l++;
                while (l < r && nums[r] == nums[r-1]) r--;
                l++; r--;
            } else if (sum < 0) l++;
            else r--;
        }
    }
    return res;
}
```

## 四数之和

```javascript
function fourSum(nums, target) {
  nums.sort((a, b) => a - b);
  const res = [];

  for (let i = 0; i < nums.length - 3; i++) {
    if (i > 0 && nums[i] === nums[i - 1]) continue;
    for (let j = i + 1; j < nums.length - 2; j++) {
      if (j > i + 1 && nums[j] === nums[j - 1]) continue;
      let l = j + 1, r = nums.length - 1;
      while (l < r) {
        const sum = nums[i] + nums[j] + nums[l] + nums[r];
        if (sum === target) {
          res.push([nums[i], nums[j], nums[l], nums[r]]);
          while (l < r && nums[l] === nums[l + 1]) l++;
          while (l < r && nums[r] === nums[r - 1]) r--;
          l++; r--;
        } else if (sum < target) l++;
        else r--;
      }
    }
  }
  return res;
}
```

## 最接近的三数之和

```javascript
function threeSumClosest(nums, target) {
  nums.sort((a, b) => a - b);
  let closest = nums[0] + nums[1] + nums[2];

  for (let i = 0; i < nums.length - 2; i++) {
    let l = i + 1, r = nums.length - 1;
    while (l < r) {
      const sum = nums[i] + nums[l] + nums[r];
      if (Math.abs(sum - target) < Math.abs(closest - target)) {
        closest = sum;
      }
      if (sum < target) l++;
      else if (sum > target) r--;
      else return sum; // 精确匹配
    }
  }
  return closest;
}
```

## 复杂度分析

| 方法 | 时间 | 空间 |
|------|------|------|
| 暴力三重循环 | O(n³) | O(1) |
| 排序+双指针 | O(n²) | O(1) 排序栈空间 |
| 哈希表法 | O(n²) | O(n) |

## 关键技巧

1. **排序预处理**：使双指针法可行，同时方便去重
2. **三层去重**：i、l、r 三个位置都需要去重
3. **剪枝优化**：`nums[i] > 0` 时可提前结束
4. **双指针移动**：和太小则 l++，和太大则 r--

## 常见陷阱

1. **去重遗漏**：三个位置的去重都不能少
2. **索引越界**：`nums.length - 2` 的边界
3. **l 和 r 的初始化**：`l = i + 1` 不是 `l = 0`
4. **排序后索引变化**：如果需要返回原索引，要先保存
