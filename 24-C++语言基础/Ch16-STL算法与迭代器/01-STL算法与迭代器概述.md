# STL算法与迭代器概述

## 一、概念说明

STL的三大组件：**容器**（存储数据）、**迭代器**（遍历数据）、**算法**（处理数据，C++标准 §25-26）。迭代器是容器和算法之间的桥梁——算法通过迭代器操作容器数据，不依赖具体容器类型。这种解耦设计是泛型编程的核心。

### 1.1 迭代器抽象的意义

```
算法 ──→ 迭代器接口 ──→ 容器
  │                        │
  │  不关心容器类型         │  不关心算法细节
  │  只关心迭代器能力       │  只提供迭代器

结果：同一算法可处理不同容器，同一容器可配合不同算法
```

## 二、三者关系

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> vec = {5, 3, 1, 4, 2};

    // 容器：存储数据
    // 迭代器：vec.begin(), vec.end()
    // 算法：std::sort
    std::sort(vec.begin(), vec.end());

    for (int v : vec) std::cout << v << " ";  // 1 2 3 4 5
}
```

## 三、算法分类

```cpp
/*
| 类别         | 示例                                   | 数量  |
|-------------|---------------------------------------|-------|
| 非修改算法    | find, count, for_each, equal          | ~20   |
| 修改算法      | copy, transform, replace, fill        | ~30   |
| 排序算法      | sort, stable_sort, partial_sort       | ~15   |
| 数值算法      | accumulate, reduce, inner_product     | ~10   |
| 集合算法      | set_union, set_intersection           | ~5    |
| 堆算法        | make_heap, push_heap, pop_heap        | ~5    |
| 最小最大      | min, max, minmax, clamp               | ~10   |
| C++20 ranges | ranges::sort, views::filter           | ~20   |
*/
```

## 四、迭代器类别

```cpp
/*
| 类别              | 能力                       | 容器示例           |
|-------------------|---------------------------|-------------------|
| 输入迭代器        | 只读，单遍                 | istream_iterator  |
| 输出迭代器        | 只写，单遍                 | ostream_iterator  |
| 前向迭代器        | 读写，多遍                 | forward_list      |
| 双向迭代器        | 可--                       | list, set, map    |
| 随机访问迭代器    | 可+/-/[]                   | vector, deque     |
| 连续迭代器(C++20) | 内存连续                    | vector, array     |
*/
```

## 五、Lambda与算法

```cpp
#include <algorithm>
#include <vector>

void lambda_with_algorithms() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // lambda作为谓词
    auto count = std::count_if(v.begin(), v.end(),
        [](int x) { return x > 3; });  // 2

    // lambda作为变换函数
    std::transform(v.begin(), v.end(), v.begin(),
        [](int x) { return x * x; });  // {1, 4, 9, 16, 25}

    // lambda捕获外部变量
    int threshold = 10;
    std::erase_if(v, [threshold](int x) { return x < threshold; });
}
```

## 六、注意事项

1. **算法不改变容器大小**：需要insert/erase等操作配合
2. **算法依赖迭代器类别**：不支持的操作会退化为低效实现
3. **lambda表达式是现代C++中提供谓词的主要方式**
4. **函数对象（仿函数）也可以作为谓词传递给算法**
5. **C++20 ranges使算法更简洁**：直接接受范围，支持投影
6. **并行算法（C++17）利用多线程加速**：大数据量时有效
