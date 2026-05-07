# ranges视图

## 一、概念说明

Ranges视图（View）是惰性求值的数据转换管道。视图不拥有数据，只提供转换后的访问接口。`std::views`命名空间提供了多种预定义视图。

## 二、具体用法

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // filter：过滤
    auto evens = v | std::views::filter([](int x) { return x % 2 == 0; });
    for (int x : evens) std::cout << x << " ";  // 2 4 6 8 10

    // transform：变换
    auto squares = v | std::views::transform([](int x) { return x * x; });

    // take：取前n个
    auto first5 = v | std::views::take(5);
    for (int x : first5) std::cout << x << " ";  // 1 2 3 4 5

    // drop：跳过前n个
    auto skip3 = v | std::views::drop(3);

    // reverse：反转
    auto reversed = v | std::views::reverse;

    // 组合管道
    auto result = v
        | std::views::filter([](int x) { return x > 3; })
        | std::views::transform([](int x) { return x * 2; })
        | std::views::take(3);
    for (int x : result) std::cout << x << " ";
    // 8 10 12
}
```

### 2.1 字符串视图

```cpp
#include <string_view>

void string_views() {
    std::string s = "hello world";

    // split：分割
    auto words = s | std::views::split(' ');
    for (auto word : words) {
        std::string_view sv(word.begin(), word.end());
        std::cout << sv << " ";
    }
    // hello world

    // join：连接
    auto joined = std::views::join(words);
}
```

## 三、注意事项

- 视图是惰性的，遍历时才计算
- 视图不拥有数据，原始数据必须有效
- 视图可以组合成复杂的管道
- 视图通常是O(1)空间复杂度
