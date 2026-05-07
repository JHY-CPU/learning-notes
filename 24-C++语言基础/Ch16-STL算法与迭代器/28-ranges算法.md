# ranges算法

## 一、概念说明

`std::ranges`命名空间提供了算法的Ranges版本，直接接受范围（而非迭代器对），支持投影，并返回迭代器+哨位对。

## 二、具体用法

```cpp
#include <ranges>
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {5, 3, 1, 4, 2};

    // ranges::sort：直接接受范围
    std::ranges::sort(v);

    // ranges::find
    auto it = std::ranges::find(v, 3);
    if (it != v.end()) std::cout << "找到: " << *it << std::endl;

    // ranges::for_each
    std::ranges::for_each(v, [](int x) { std::cout << x << " "; });

    // ranges::all_of / any_of / none_of
    bool all_positive = std::ranges::all_of(v, [](int x) { return x > 0; });

    // ranges::copy
    std::vector<int> dst(v.size());
    std::ranges::copy(v, dst.begin());

    // ranges::count
    auto cnt = std::ranges::count(v, 3);

    // 投影
    struct Point { int x, y; };
    std::vector<Point> points = {{1, 2}, {3, 4}, {5, 1}};
    std::ranges::sort(points, {}, &Point::y);  // 按y排序
}
```

## 三、注意事项

- ranges算法直接接受范围，更简洁
- 支持投影，无需手动提取字段
- 返回迭代器+哨位对（非单个迭代器）
- 与传统算法并存，可以混用
