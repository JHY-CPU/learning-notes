# ranges视图

## 一、概念说明

Ranges视图（C++标准 §24.6-§24.7）是惰性求值的数据转换管道。视图不拥有数据，只提供转换后的访问接口。`std::views`命名空间提供了多种预定义视图。视图的核心优势是**零拷贝**和**惰性求值**。

### 1.1 常用视图

| 视图 | 功能 |
|------|------|
| `filter` | 过滤 |
| `transform` | 变换 |
| `take` | 取前n个 |
| `drop` | 跳过前n个 |
| `reverse` | 反转 |
| `split` | 分割 |
| `join` | 连接 |
| `iota` | 生成序列 |

## 二、具体用法

```cpp
#include <ranges>
#include <vector>
#include <iostream>
#include <string>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // filter：过滤
    auto evens = v | std::views::filter([](int x) { return x % 2 == 0; });
    for (int x : evens) std::cout << x << " ";  // 2 4 6 8 10

    // transform：变换
    auto squares = v | std::views::transform([](int x) { return x * x; });

    // take：取前n个
    auto first5 = v | std::views::take(5);

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

### 2.1 iota视图

```cpp
void iota_demo() {
    // 生成序列 [1, 10]
    auto seq = std::views::iota(1, 11);
    for (int x : seq) std::cout << x << " ";  // 1 2 3 ... 10

    // 无限序列
    auto infinite = std::views::iota(0);
    auto first10 = infinite | std::views::take(10);
}
```

### 2.2 split和join

```cpp
#include <string_view>

void split_join() {
    std::string s = "hello world foo bar";

    // split：分割
    auto words = s | std::views::split(' ');
    for (auto word : words) {
        std::string_view sv(word.begin(), word.end());
        std::cout << sv << " ";
    }

    // join：连接
    std::vector<std::vector<int>> nested = {{1, 2}, {3, 4}, {5, 6}};
    auto flat = nested | std::views::join;
    for (int x : flat) std::cout << x << " ";  // 1 2 3 4 5 6
}
```

## 三、注意事项

1. **视图是惰性的**：遍历时才计算
2. **视图不拥有数据**：原始数据必须有效
3. **视图可以组合成复杂的管道**
4. **视图通常是O(1)空间复杂度**
5. **`views::iota`可生成无限序列**：配合`take`使用
6. **`split`返回的不是string**：是范围的范围
