# 排序与搜索专题 (Sorting & Searching)

## 一、排序算法总结

### 1.1 常见排序算法对比

| 算法 | 平均时间 | 最坏时间 | 空间 | 稳定性 | 适用场景 |
|------|---------|---------|------|--------|---------|
| 冒泡排序 | $O(n^2)$ | $O(n^2)$ | $O(1)$ | 稳定 | 教学用 |
| 选择排序 | $O(n^2)$ | $O(n^2)$ | $O(1)$ | 不稳定 | 小数据 |
| 插入排序 | $O(n^2)$ | $O(n^2)$ | $O(1)$ | 稳定 | 近乎有序 |
| 归并排序 | $O(n\log n)$ | $O(n\log n)$ | $O(n)$ | 稳定 | 链表排序 |
| 快速排序 | $O(n\log n)$ | $O(n^2)$ | $O(\log n)$ | 不稳定 | 通用首选 |
| 堆排序 | $O(n\log n)$ | $O(n\log n)$ | $O(1)$ | 不稳定 | TopK问题 |
| 计数排序 | $O(n+k)$ | $O(n+k)$ | $O(k)$ | 稳定 | 范围已知 |
| 桶排序 | $O(n+k)$ | $O(n^2)$ | $O(n+k)$ | 稳定 | 均匀分布 |
| 基数排序 | $O(dn)$ | $O(dn)$ | $O(n+k)$ | 稳定 | 整数/字符串 |

### 1.2 快速排序

```python
def quick_sort(arr, left, right):
    if left >= right: return
    pivot = arr[right]
    i = left
    for j in range(left, right):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[right] = arr[right], arr[i]
    quick_sort(arr, left, i - 1)
    quick_sort(arr, i + 1, right)
```

**随机化优化：** 随机选择pivot，避免最坏情况。

```cpp
void quickSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int pivotIdx = left + rand() % (right - left + 1);
    swap(arr[pivotIdx], arr[right]);
    int pivot = arr[right], i = left;
    for (int j = left; j < right; j++) {
        if (arr[j] < pivot) swap(arr[i++], arr[j]);
    }
    swap(arr[i], arr[right]);
    quickSort(arr, left, i - 1);
    quickSort(arr, i + 1, right);
}
```

### 1.3 归并排序

```python
def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### 1.4 堆排序

```python
def heap_sort(arr):
    n = len(arr)

    def sift_down(i, size):
        largest = i
        left, right = 2*i+1, 2*i+2
        if left < size and arr[left] > arr[largest]: largest = left
        if right < size and arr[right] > arr[largest]: largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            sift_down(largest, size)

    # 建堆
    for i in range(n//2-1, -1, -1):
        sift_down(i, n)
    # 排序
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        sift_down(0, i)
```

---

## 二、二分查找

### 2.1 标准二分

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target: return mid
        elif nums[mid] < target: left = mid + 1
        else: right = mid - 1
    return -1
```

### 2.2 查找边界

```python
# 第一个 >= target 的位置
def lower_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target: left = mid + 1
        else: right = mid
    return left

# 第一个 > target 的位置
def upper_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] <= target: left = mid + 1
        else: right = mid
    return left
```

### 2.3 二分答案

适用于：答案具有单调性，可以直接验证某个值是否可行。

```python
# LeetCode 875: 爱吃香蕉的珂珂
def min_eating_speed(piles, h):
    left, right = 1, max(piles)
    while left < right:
        mid = left + (right - left) // 2
        hours = sum((p + mid - 1) // mid for p in piles)
        if hours <= h: right = mid
        else: left = mid + 1
    return left
```

### 2.4 旋转数组搜索 (LeetCode 33)

```python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target: return mid
        if nums[left] <= nums[mid]:  # 左半有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # 右半有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

---

## 三、TopK问题

### 3.1 快速选择算法

```python
import random

def find_kth_largest(nums, k):
    target = len(nums) - k

    def quick_select(left, right):
        pivot_idx = random.randint(left, right)
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        pivot = nums[right]
        i = left
        for j in range(left, right):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]

        if i == target: return nums[i]
        elif i < target: return quick_select(i + 1, right)
        else: return quick_select(left, i - 1)

    return quick_select(0, len(nums) - 1)
```

### 3.2 堆方法

```python
import heapq

def find_kth_largest_heap(nums, k):
    return heapq.nlargest(k, nums)[-1]
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 快速排序 | $O(n\log n)$ 平均 | $O(\log n)$ |
| 归并排序 | $O(n\log n)$ | $O(n)$ |
| 堆排序 | $O(n\log n)$ | $O(1)$ |
| 二分查找 | $O(\log n)$ | $O(1)$ |
| 快速选择 | $O(n)$ 平均 | $O(1)$ |

---

## 五、面试高频题

1. **LeetCode 912：** 排序数组
2. **LeetCode 33：** 搜索旋转排序数组
3. **LeetCode 34：** 在排序数组中查找元素的第一个和最后一个位置
4. **LeetCode 215：** 数组中的第K个最大元素
5. **LeetCode 875：** 爱吃香蕉的珂珂
6. **LeetCode 4：** 寻找两个正序数组的中位数
7. **LeetCode 153：** 寻找旋转排序数组中的最小值
8. **LeetCode 74：** 搜索二维矩阵
9. **LeetCode 347：** 前K个高频元素
10. **LeetCode 973：** 最接近原点的K个点
