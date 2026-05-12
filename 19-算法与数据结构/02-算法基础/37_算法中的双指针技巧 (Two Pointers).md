# 38-算法中的双指针技巧 (Two Pointers)

双指针是用两个指针协同遍历数据结构的技术，能将某些 O(n²) 问题优化到 O(n)。是数组和链表问题的核心技巧。

## 三种模式

| 模式 | 指针方向 | 适用场景 |
|------|---------|---------|
| 对撞指针 | 左右向中间 | 有序数组、回文、容器 |
| 快慢指针 | 同向不同速 | 链表环、中点 |
| 分离双指针 | 两个独立数组 | 合并、归并 |

## JavaScript 实现

```javascript
// 1. 对撞指针：有序数组两数之和（LeetCode 167）
function twoSumSorted(nums, target) {
  let l = 0, r = nums.length - 1;
  while (l < r) {
    const sum = nums[l] + nums[r];
    if (sum === target) return [l + 1, r + 1];
    if (sum < target) l++;
    else r--;
  }
  return [-1, -1];
}

// 2. 对撞指针：盛最多水的容器（LeetCode 11）
function maxArea(height) {
  let l = 0, r = height.length - 1, max = 0;
  while (l < r) {
    const area = Math.min(height[l], height[r]) * (r - l);
    max = Math.max(max, area);
    if (height[l] < height[r]) l++;
    else r--;
  }
  return max;
}

// 3. 对撞指针：验证回文串（LeetCode 125）
function isPalindrome(s) {
  s = s.toLowerCase().replace(/[^a-z0-9]/g, '');
  let l = 0, r = s.length - 1;
  while (l < r) {
    if (s[l] !== s[r]) return false;
    l++; r--;
  }
  return true;
}

// 4. 快慢指针：链表环检测（LeetCode 141）
function hasCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
}

// 5. 快慢指针：删除有序数组重复项（LeetCode 26）
function removeDuplicates(nums) {
  if (nums.length === 0) return 0;
  let slow = 0;
  for (let fast = 1; fast < nums.length; fast++) {
    if (nums[fast] !== nums[slow]) {
      slow++;
      nums[slow] = nums[fast];
    }
  }
  return slow + 1;
}

// 6. 分离双指针：合并两个有序数组
function mergeSorted(arr1, arr2) {
  const result = [];
  let i = 0, j = 0;
  while (i < arr1.length && j < arr2.length) {
    if (arr1[i] <= arr2[j]) result.push(arr1[i++]);
    else result.push(arr2[j++]);
  }
  while (i < arr1.length) result.push(arr1[i++]);
  while (j < arr2.length) result.push(arr2[j++]);
  return result;
}

// 7. 三数之和（LeetCode 15）：排序 + 双指针
function threeSum(nums) {
  nums.sort((a, b) => a - b);
  const result = [];
  for (let i = 0; i < nums.length - 2; i++) {
    if (i > 0 && nums[i] === nums[i - 1]) continue;
    let l = i + 1, r = nums.length - 1;
    while (l < r) {
      const sum = nums[i] + nums[l] + nums[r];
      if (sum === 0) {
        result.push([nums[i], nums[l], nums[r]]);
        while (l < r && nums[l] === nums[l + 1]) l++;
        while (l < r && nums[r] === nums[r - 1]) r--;
        l++; r--;
      } else if (sum < 0) l++;
      else r--;
    }
  }
  return result;
}

// 测试
console.log(twoSumSorted([2, 7, 11, 15], 9));     // [1, 2]
console.log(maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7])); // 49
console.log(isPalindrome("A man, a plan, a canal: Panama")); // true
console.log(removeDuplicates([1, 1, 2]));            // 2
console.log(threeSum([-1, 0, 1, 2, -1, -4]));       // [[-1,-1,2],[-1,0,1]]
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
#include <string>
using namespace std;

// 两数之和 II
vector<int> twoSumSorted(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        int sum = nums[l] + nums[r];
        if (sum == target) return {l + 1, r + 1};
        if (sum < target) l++;
        else r--;
    }
    return {};
}

// 三数之和
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> result;
    for (int i = 0; i < (int)nums.size() - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        int l = i + 1, r = nums.size() - 1;
        while (l < r) {
            int sum = nums[i] + nums[l] + nums[r];
            if (sum == 0) {
                result.push_back({nums[i], nums[l], nums[r]});
                while (l < r && nums[l] == nums[l + 1]) l++;
                while (l < r && nums[r] == nums[r - 1]) r--;
                l++; r--;
            } else if (sum < 0) l++;
            else r--;
        }
    }
    return result;
}

// 盛最多水的容器
int maxArea(vector<int>& height) {
    int l = 0, r = height.size() - 1, maxA = 0;
    while (l < r) {
        maxA = max(maxA, min(height[l], height[r]) * (r - l));
        if (height[l] < height[r]) l++;
        else r--;
    }
    return maxA;
}

// 删除有序数组重复项
int removeDuplicates(vector<int>& nums) {
    if (nums.empty()) return 0;
    int slow = 0;
    for (int fast = 1; fast < nums.size(); fast++) {
        if (nums[fast] != nums[slow]) nums[++slow] = nums[fast];
    }
    return slow + 1;
}
```

## 复杂度

| 算法 | 时间 | 空间 |
|------|------|------|
| 两数之和（有序） | O(n) | O(1) |
| 三数之和 | O(n²) | O(1) |
| 盛最多水 | O(n) | O(1) |
| 链表环检测 | O(n) | O(1) |
| 删除重复项 | O(n) | O(1) |
| 合并有序数组 | O(n + m) | O(n + m) |

## 常见陷阱

1. **去重遗漏**：三数之和中内外层循环都需要去重
2. **边界条件**：空数组、单元素、全相同元素
3. **快慢指针起点**：slow 从 0 开始，fast 从 1 开始（删除重复项）
4. **while 条件**：`l < r` vs `l <= r` 根据问题选择

## 实际应用

双指针是面试中出现频率最高的技巧之一。LeetCode 11、15、167、141、26 都是其经典应用。遇到有序数组问题或需要线性时间优化的问题，优先考虑双指针。
