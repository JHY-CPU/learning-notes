## 11-滑动窗口基础 (Sliding Window Basics)

滑动窗口是一种通过维护一个可变或固定大小的窗口来简化问题的方法，将嵌套循环优化为单循环。

## 固定窗口 vs 可变窗口

```javascript

// 固定窗口大小 - 在数组上滑动固定长度的窗口
function fixedWindow(arr, k) {
  // 计算第一个窗口
  let windowSum = 0;
  for (let i = 0; i < k; i++) {
    windowSum += arr[i];
  }
  let maxSum = windowSum;

  // 滑动窗口：移除左边，加入右边
  for (let i = k; i < arr.length; i++) {
    windowSum += arr[i] - arr[i - k];
    maxSum = Math.max(maxSum, windowSum);
  }

  return maxSum;
}

// 可变窗口大小 - 根据条件动态调整
function variableWindow(arr, target) {
  let left = 0;
  let windowSum = 0;
  let minLen = Infinity;

  for (let right = 0; right < arr.length; right++) {
    windowSum += arr[right]; // 扩展右边界

    while (windowSum >= target) { // 满足条件时收缩左边界
      minLen = Math.min(minLen, right - left + 1);
      windowSum -= arr[left];
      left++;
    }
  }

  return minLen === Infinity ? 0 : minLen;
}
```

## 滑动窗口模板

```javascript

// 通用滑动窗口模板
function slidingWindow(s) {
  let left = 0, right = 0;
  let window = {};

  while (right < s.length) {
    // 扩大窗口
    let char = s[right];
    window[char] = (window[char] || 0) + 1;
    right++;

    // 满足条件时收缩
    while (needShrink(window)) {
      let d = s[left];
      window[d]--;
      if (window[d] === 0) delete window[d];
      left++;
    }
  }
}
```

## 交互演示
