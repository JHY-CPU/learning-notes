## 09-对撞指针 (Colliding Pointers)

对撞指针（相向双指针）是指两个指针从数组两端向中间移动，常用于有序数组的搜索和反转问题。

## 对撞指针模式

```javascript

// 对撞指针基本模板：
function collidingPointers(arr, target) {
  let left = 0;
  let right = arr.length - 1;

  while (left < right) {
    // 计算当前状态
    let current = compute(left, right);

    if (current === target) {
      return [left, right]; // 找到答案
    } else if (current < target) {
      left++; // 左指针右移
    } else {
      right--; // 右指针左移
    }
  }

  return [-1, -1]; // 未找到
}
```

## 经典例题

```javascript

// 1. 两数之和 II（有序数组）
function twoSum(numbers, target) {
  let l = 0, r = numbers.length - 1;
  while (l < r) {
    let sum = numbers[l] + numbers[r];
    if (sum === target) return [l + 1, r + 1];
    if (sum < target) l++;
    else r--;
  }
  return [-1, -1];
}

// 2. 验证回文串
function isPalindrome(s) {
  s = s.toLowerCase().replace(/[^a-z0-9]/g, '');
  let l = 0, r = s.length - 1;
  while (l < r) {
    if (s[l] !== s[r]) return false;
    l++;
    r--;
  }
  return true;
}

// 3. 盛最多水的容器
function maxArea(height) {
  let l = 0, r = height.length - 1;
  let max = 0;
  while (l < r) {
    let area = Math.min(height[l], height[r]) * (r - l);
    max = Math.max(max, area);
    if (height[l] < height[r]) l++;
    else r--;
  }
  return max;
}
```

## 交互演示
