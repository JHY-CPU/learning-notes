# 10-快慢指针 (Fast & Slow Pointers)

快慢指针是双指针的一种，两个指针从同一起点出发，但移动速度不同。常用于链表环检测、寻找中点等场景。

## 快慢指针核心思想

```javascript
// 快指针每次走2步，慢指针每次走1步
// 快指针到末尾时，慢指针在中间位置
function fastSlowPointer(arr) {
  let slow = 0;
  let fast = 0;
  while (fast < arr.length && fast + 1 < arr.length) {
    slow++;
    fast += 2;
  }
  return slow;
}
```

## C++ 实现

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// 寻找链表中间节点
ListNode* findMiddle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}

// 检测链表环
bool hasCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}

// 找环的入口
ListNode* detectCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            ListNode* p = head;
            while (p != slow) {
                p = p->next;
                slow = slow->next;
            }
            return p;
        }
    }
    return nullptr;
}
```

## 常见应用

```javascript
// 1. 寻找链表中间节点
function findMiddle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  return slow;
}

// 2. 检测链表环
function hasCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
}

// 3. 找环的入口
function detectCycle(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) {
      let p = head;
      while (p !== slow) { p = p.next; slow = slow.next; }
      return p;
    }
  }
  return null;
}

// 4. 移除有序数组中的重复元素
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

// 5. 判断链表是否回文
function isPalindrome(head) {
  // 找中点
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  // 反转后半部分
  let prev = null;
  while (slow) {
    let next = slow.next;
    slow.next = prev;
    prev = slow;
    slow = next;
  }
  // 比较
  let p1 = head, p2 = prev;
  while (p2) {
    if (p1.val !== p2.val) return false;
    p1 = p1.next;
    p2 = p2.next;
  }
  return true;
}
```

## 快慢指针检测环的原理

设环长为 L，慢指针入环后走了 k 步与快指针相遇：
- 快指针走过的距离是慢指针的 2 倍
- 相遇时快指针比慢指针多走 n 圈环
- 即：2(slow) = slow + nL => slow = nL
- 从相遇点到入口的距离 = L - k，从头到入口 = 环入口距离
- 让一个指针从头走，一个从相遇点走，必在入口相遇

## 时间复杂度

所有快慢指针操作都是 O(n) 时间 O(1) 空间。

## 常见陷阱

1. 空链表或单节点链表的边界处理
2. `fast.next` 和 `fast.next.next` 的空检查顺序
3. 找中点时偶数长度链表的中点选择（靠前还是靠后）
4. 检测环时注意快慢指针的初始位置（都从 head 开始）
