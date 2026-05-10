# 排序专题 (Sorting Problems)

## 一、概念定义与原理

### 1.1 常见排序算法

| 算法 | 平均复杂度 | 最坏复杂度 | 空间 | 稳定性 |
|------|-----------|-----------|------|--------|
| 冒泡排序 | $O(n^2)$ | $O(n^2)$ | $O(1)$ | 稳定 |
| 选择排序 | $O(n^2)$ | $O(n^2)$ | $O(1)$ | 不稳定 |
| 插入排序 | $O(n^2)$ | $O(n^2)$ | $O(1)$ | 稳定 |
| 归并排序 | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | 稳定 |
| 快速排序 | $O(n \log n)$ | $O(n^2)$ | $O(\log n)$ | 不稳定 |
| 堆排序 | $O(n \log n)$ | $O(n \log n)$ | $O(1)$ | 不稳定 |
| 计数排序 | $O(n+k)$ | $O(n+k)$ | $O(k)$ | 稳定 |

### 1.2 排序的竞赛应用

- 排序后对撞指针/二分查找
- 自定义排序规则
- 逆序对计数（归并排序）
- 第K大元素（快速选择）

---

## 二、核心算法

### 2.1 快速排序

```cpp
void quicksort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int pivot = a[l + rand() % (r - l + 1)];
    int i = l, j = r;
    while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) swap(a[i++], a[j--]);
    }
    quicksort(a, l, j);
    quicksort(a, i, r);
}
```

### 2.2 归并排序（含逆序对计数）

```cpp
long long merge_count(vector<int>& a, int l, int r) {
    if (l >= r) return 0;
    int m = l + (r - l) / 2;
    long long cnt = merge_count(a, l, m) + merge_count(a, m+1, r);
    vector<int> temp;
    int i = l, j = m + 1;
    while (i <= m && j <= r) {
        if (a[i] <= a[j]) temp.push_back(a[i++]);
        else { temp.push_back(a[j++]); cnt += m - i + 1; }
    }
    while (i <= m) temp.push_back(a[i++]);
    while (j <= r) temp.push_back(a[j++]);
    copy(temp.begin(), temp.end(), a.begin() + l);
    return cnt;
}
```

### 2.3 快速选择（第K大）

```cpp
int quickselect(vector<int>& a, int l, int r, int k) {
    if (l == r) return a[l];
    int pivot = a[l + rand() % (r - l + 1)];
    int i = l, j = r;
    while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) swap(a[i++], a[j--]);
    }
    if (k <= j) return quickselect(a, l, j, k);
    return quickselect(a, i, r, k);
}
```

---

## 三、代码实现

### 3.1 堆排序 - C++

```cpp
void heapify(vector<int>& a, int n, int i) {
    int largest = i, l = 2*i+1, r = 2*i+2;
    if (l < n && a[l] > a[largest]) largest = l;
    if (r < n && a[r] > a[largest]) largest = r;
    if (largest != i) {
        swap(a[i], a[largest]);
        heapify(a, n, largest);
    }
}

void heapsort(vector<int>& a) {
    int n = a.size();
    for (int i = n/2-1; i >= 0; i--) heapify(a, n, i);
    for (int i = n-1; i > 0; i--) {
        swap(a[0], a[i]);
        heapify(a, i, 0);
    }
}
```

### 3.2 自定义排序 - C++

```cpp
// 按绝对值排序
sort(a.begin(), a.end(), [](int x, int y) { return abs(x) < abs(y); });

// 按多个关键字排序
sort(students.begin(), students.end(), [](const auto& a, const auto& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.name < b.name;
});

// 按到原点距离排序
sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
    return a.x*a.x + a.y*a.y < b.x*b.x + b.y*b.y;
});
```

### 3.3 Python 实现

```python
def quicksort(a, l, r):
    if l >= r: return
    pivot = a[l]; i, j = l, r
    while i <= j:
        while i <= j and a[i] < pivot: i += 1
        while i <= j and a[j] > pivot: j -= 1
        if i <= j: a[i], a[j] = a[j], a[i]; i += 1; j -= 1
    quicksort(a, l, j); quicksort(a, i, r)

def merge_count(a):
    if len(a) <= 1: return a, 0
    mid = len(a) // 2
    left, cnt_l = merge_count(a[:mid])
    right, cnt_r = merge_count(a[mid:])
    merged, cnt = [], cnt_l + cnt_r
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]: merged.append(left[i]); i += 1
        else: merged.append(right[j]); j += 1; cnt += len(left) - i
    merged += left[i:] + right[j:]
    return merged, cnt

print(merge_count([5,4,3,2,1]))  # ([1,2,3,4,5], 10)

# 自定义排序
students = [("Alice", 90), ("Bob", 85), ("Charlie", 90)]
students.sort(key=lambda x: (-x[1], x[0]))
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| STL sort | $O(n \log n)$ | Introsort（快排+堆排+插入） |
| 逆序对计数 | $O(n \log n)$ | 归并排序 |
| 第K大 | $O(n)$ 期望 | 快速选择 |
| 计数排序 | $O(n+k)$ | 值域有限时 |

---

## 五、竞赛与面试应用场景

1. **LeetCode 912：** 排序数组
2. **LeetCode 215：** 数组中的第K个最大元素
3. **逆序对：** 归并排序求解
4. **区间调度：** 按结束时间排序
5. **自定义排序：** 各种比较规则
