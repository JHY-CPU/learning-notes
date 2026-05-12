# 29-排序稳定性分析 (Sort Stability)

排序稳定性指：相等元素在排序后保持原有的相对顺序。

## 稳定 vs 不稳定

| 稳定排序 | 不稳定排序 |
|---------|-----------|
| 冒泡排序 | 选择排序 |
| 插入排序 | 快速排序 |
| 归并排序 | 堆排序 |
| 计数排序 | 希尔排序 |
| 桶排序 | |
| 基数排序 | |

判断技巧：排序过程中如果有"不相邻元素的交换"，则不稳定。

## JavaScript 实现

```javascript
// 演示稳定性的重要性
const students = [
  { name: 'Alice', grade: 85 },
  { name: 'Bob', grade: 90 },
  { name: 'Charlie', grade: 85 },
  { name: 'David', grade: 80 }
];

// 稳定排序：相同 grade 保持原顺序
function stableSortDemo() {
  const sorted = [...students].sort((a, b) => a.grade - b.grade);
  console.log(sorted.map(s => `${s.name}:${s.grade}`));
  // David:80, Alice:85, Charlie:85, Bob:90
  // Alice 和 Charlie 保持原有顺序
}

// 将不稳定排序变稳定：加原始下标作为第二关键字
function makeStable(sortFn, arr, compare) {
  const indexed = arr.map((val, idx) => ({ val, idx }));
  sortFn(indexed, (a, b) => {
    const cmp = compare(a.val, b.val);
    return cmp !== 0 ? cmp : a.idx - b.idx;
  });
  return indexed.map(x => x.val);
}

// 演示选择排序的不稳定性
function unstableDemo() {
  const data = [
    { v: 3, id: 'a' },
    { v: 1, id: 'b' },
    { v: 3, id: 'c' },
    { v: 2, id: 'd' }
  ];
  // 选择排序（不稳定）
  for (let i = 0; i < data.length - 1; i++) {
    let minIdx = i;
    for (let j = i + 1; j < data.length; j++) {
      if (data[j].v < data[minIdx].v) minIdx = j;
    }
    if (minIdx !== i) [data[i], data[minIdx]] = [data[minIdx], data[i]];
  }
  // v=3 的 a 和 c 可能不再保持原顺序
  const v3 = data.filter(x => x.v === 3).map(x => x.id);
  console.log('不稳定结果:', v3); // 可能是 ['c', 'a']

  // 加下标使其稳定
  const data2 = data.map((x, i) => ({ ...x, origIdx: i }));
  // ... 使用稳定排序
}

stableSortDemo();
unstableDemo();
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

struct Student { string name; int grade; };

void demo() {
    vector<Student> students = {{"Alice",85}, {"Bob",90}, {"Charlie",85}};
    // stable_sort 保持相等元素顺序
    stable_sort(students.begin(), students.end(),
        [](const Student& a, const Student& b) { return a.grade < b.grade; });
    // Alice 和 Charlie 保持原顺序
}
```

## 为什么稳定性重要

| 场景 | 说明 |
|------|------|
| 多键排序 | 先按次要键排序，再按主要键稳定排序 |
| 数据库 ORDER BY | ORDER BY grade, name 需要稳定性 |
| UI 列表 | 保持用户之前的排序偏好 |
| 基数排序 | 必须使用稳定排序作为子程序 |

## 常见陷阱

1. **JS Array.sort**：规范不保证稳定（但 V8 实际使用 TimSort，是稳定的）
2. **Java Arrays.sort**：基本类型不稳定（双轴快排），对象类型稳定（TimSort）
3. **误用不稳定排序**：多键排序时结果不确定
