## 08-双指针技术 (Two Pointers Technique)

双指针技术使用两个指针遍历数组（通常同向或相向移动），将 O(n^2) 的暴力解法优化到 O(n)。

## 双指针核心思想

使用两个指针变量，而不是一个，来同时处理数组。常见的模式有：

```javascript

// 模式1：同向双指针（快慢指针）
// 两指针从同一端出发，但速度不同
function removeDuplicates(nums) {
  if (nums.length === 0) return 0;
  let slow = 0; // 慢指针指向不重复区域的末尾
  for (let fast = 1; fast < nums.length; fast++) {
    if (nums[fast] !== nums[slow]) {
      slow++;
      nums[slow] = nums[fast];
    }
  }
  return slow + 1; // 新长度
}

// 模式2：相向双指针（对撞指针）
// 两指针从两端向中间移动
function twoSum(nums, target) {
  let left = 0, right = nums.length - 1;
  while (left < right) {
    let sum = nums[left] + nums[right];
    if (sum === target) return [left, right];
    if (sum < target) left++;
    else right--;
  }
  return [-1, -1];
}
```

## 经典应用

```javascript

// 1. 有序数组的两数之和
// 2. 移除指定元素
// 3. 移动零到末尾
// 4. 连续子数组求和
// 5. 奇偶排序（奇数在前偶数在后）
// 6. 三数之和

// 移动零：将数组中所有0移到末尾，保持非零元素相对顺序
function moveZeroes(nums) {
  let slow = 0;
  for (let fast = 0; fast < nums.length; fast++) {
    if (nums[fast] !== 0) {
      [nums[slow], nums[fast]] = [nums[fast], nums[slow]];
      slow++;
    }
  }
}
```

## 交互演示
