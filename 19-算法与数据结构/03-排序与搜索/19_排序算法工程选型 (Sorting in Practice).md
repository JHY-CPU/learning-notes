# 20-排序算法工程选型 (Sorting in Practice)

在实际工程中，选择排序算法需要考虑数据特征、稳定性、内存限制等多方面因素。

## 各语言内置排序

| 语言/平台 | 算法 | 特点 |
|-----------|------|------|
| JavaScript (V8) | TimSort | 归并+插入混合，稳定 |
| Python | TimSort | 稳定，最坏 O(n log n) |
| Java (对象) | TimSort | 稳定排序 |
| Java (基本类型) | Dual-Pivot QuickSort | 双轴快排，不稳定 |
| C++ std::sort | IntroSort | 快排+堆排+插入混合 |
| C++ stable_sort | 归并排序变体 | 稳定 |
| Rust | 迭代归并 | O(n log n)，稳定 |
| Go | pdqsort | 模式感知快排 |

## JavaScript 实现：自定义排序

```javascript
// 数字排序（JS 默认按字符串排序）
[10, 2, 1].sort();           // [1, 10, 2] 错误！
[10, 2, 1].sort((a, b) => a - b); // [1, 2, 10] 正确

// 对象多键排序
const students = [
  { name: 'Alice', grade: 85 },
  { name: 'Bob', grade: 90 },
  { name: 'Charlie', grade: 85 }
];
students.sort((a, b) => {
  if (a.grade !== b.grade) return b.grade - a.grade; // 成绩降序
  return a.name.localeCompare(b.name);              // 姓名升序
});

// 稳定排序实现
function stableSort(arr, compare) {
  return arr.map((val, idx) => ({ val, idx }))
    .sort((a, b) => {
      const cmp = compare(a.val, b.val);
      return cmp !== 0 ? cmp : a.idx - b.idx;
    })
    .map(x => x.val);
}
```

## C++ 实现：STL 排序

```cpp
#include <vector>
#include <algorithm>
#include <string>
using namespace std;

struct Student {
    string name;
    int grade;
};

void demo() {
    vector<int> v = {5, 2, 8, 1};
    sort(v.begin(), v.end());                          // 快排变体
    stable_sort(v.begin(), v.end());                   // 归并变体
    partial_sort(v.begin(), v.begin() + 2, v.end());   // 前2个最小
    nth_element(v.begin(), v.begin() + 2, v.end());    // 第3小

    // 自定义比较
    sort(v.begin(), v.end(), greater<int>());          // 降序

    // 对象排序
    vector<Student> students = {{"Alice", 85}, {"Bob", 90}};
    sort(students.begin(), students.end(), [](const Student& a, const Student& b) {
        if (a.grade != b.grade) return a.grade > b.grade;
        return a.name < b.name;
    });
}
```

## 工程选型建议

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 通用排序 | 语言内置 sort | 经过充分优化 |
| 小数组 | 插入排序 | 常数小 |
| 需要稳定 | stable_sort | 保持相对顺序 |
| 只需前K个 | partial_sort / nth_element | 比全排序快 |
| 外部数据 | 归并排序 | 顺序访问友好 |
| 内存受限 | 堆排序 | O(1) 空间 |
| 整数范围小 | 计数排序 | O(n) 时间 |

## 常见陷阱

1. **JS 数字排序**：默认按字符串排序，必须传比较函数
2. **比较函数返回值**：应返回负数/0/正数，不是 true/false
3. **不稳定性误用**：用 sort 做多键排序时结果可能不确定
4. **过度优化**：优先用内置排序，除非有明确瓶颈
