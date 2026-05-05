## 10-快慢指针 (Fast & Slow Pointers)

快慢指针是双指针的一种，两个指针从同一起点出发，但移动速度不同。常用于链表环检测、寻找中点等场景。

## 快慢指针核心思想

```javascript

// 快慢指针模板：快指针每次走2步，慢指针每次走1步
function fastSlowPointer(arr) {
  let slow = 0;
  let fast = 0;

  while (fast < arr.length) {
    // 慢指针条件移动
    // 快指针条件移动（通常是2倍速）

    slow++;      // 慢指针移动1步
    fast += 2;   // 快指针移动2步
  }

  return slow; // 通常返回慢指针位置
}
```

## 常见应用

```javascript

// 1. 寻找链表中间节点
// 快指针到末尾时，慢指针恰好在中间
function findMiddle(head) {
  let slow = head;
  let fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  return slow;
}

// 2. 检测链表环
function hasCycle(head) {
  let slow = head;
  let fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true; // 相遇则有环
  }
  return false;
}

// 3. 移除有序数组中的重复元素
function removeDuplicates(nums) {
  if (nums.length === 0) return 0;
  let slow = 0; // 不重复区域的右边界
  for (let fast = 1; fast < nums.length; fast++) {
    if (nums[fast] !== nums[slow]) {
      slow++;
      nums[slow] = nums[fast];
    }
  }
  return slow + 1;
}
```

## 交互演示
