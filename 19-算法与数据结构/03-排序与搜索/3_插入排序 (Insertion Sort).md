# 4-插入排序 (Insertion Sort)

插入排序通过构建有序序列，对未排序数据从后向前扫描，找到相应位置并插入。

## 复杂度分析

| 情况 | 时间 | 空间 |
|------|------|------|
| 最好（已排序） | O(n) | O(1) |
| 平均 | O(n²) | O(1) |
| 最坏（逆序） | O(n²) | O(1) |

稳定性：稳定。原地排序：是。

## JavaScript 实现

```javascript
// 基础插入排序
function insertionSort(arr) {
  const n = arr.length;
  for (let i = 1; i < n; i++) {
    const key = arr[i];
    let j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
  return arr;
}

// 二分插入排序（减少比较次数）
function binaryInsertionSort(arr) {
  for (let i = 1; i < arr.length; i++) {
    const key = arr[i];
    // 二分查找插入位置
    let l = 0, r = i;
    while (l < r) {
      const mid = (l + r) >> 1;
      if (arr[mid] <= key) l = mid + 1;
      else r = mid;
    }
    // 移动元素并插入
    for (let j = i; j > l; j--) arr[j] = arr[j - 1];
    arr[l] = key;
  }
  return arr;
}

console.log(insertionSort([5, 2, 4, 6, 1, 3]));       // [1, 2, 3, 4, 5, 6]
console.log(binaryInsertionSort([5, 2, 4, 6, 1, 3]));  // [1, 2, 3, 4, 5, 6]
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

void insertionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// 链表插入排序
struct ListNode {
    int val;
    ListNode* next;
};

ListNode* insertionSortList(ListNode* head) {
    ListNode dummy(0);
    while (head) {
        ListNode* next = head->next;
        ListNode* p = &dummy;
        while (p->next && p->next->val < head->val) p = p->next;
        head->next = p->next;
        p->next = head;
        head = next;
    }
    return dummy.next;
}
```

## 适用场景

- 近乎有序的数据：接近 O(n)，效率极高
- 小规模数据（n < 50）：很多标准库对小数组使用插入排序
- 在线排序：数据逐个到达时边接收边排序
- 链表排序：天然适合，只需调整指针

## 变体

| 变体 | 改进点 |
|------|--------|
| 二分插入排序 | 比较次数 O(n log n)，移动仍 O(n²) |
| 希尔排序 | 按步长分组插入，可降到 O(n^1.3) |
| 链表插入排序 | 不需移动元素，只调指针 |

## 常见陷阱

1. **循环条件**：j >= 0 且 arr[j] > key，两者都要满足
2. **性能预期**：虽然最好 O(n)，但平均仍是 O(n²)
3. **二分插入排序**：减少比较但不减少移动，总时间仍是 O(n²)
