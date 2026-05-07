# optional详解

## 一、概念说明

`std::optional`是C++17引入的包装类，表示一个值可能存在也可能不存在。它替代了使用特殊值（如-1、nullptr）表示"无值"的做法，提供类型安全的可选值语义。

## 二、具体用法

```cpp
#include <optional>
#include <iostream>
#include <string>

std::optional<int> find_index(const std::vector<int>& vec, int target) {
    for (size_t i = 0; i < vec.size(); ++i)
        if (vec[i] == target) return static_cast<int>(i);
    return std::nullopt;  // 无值
}

int main() {
    std::vector<int> v = {10, 20, 30, 40};

    auto idx = find_index(v, 30);
    if (idx) {
        std::cout << "找到: " << *idx << std::endl;  // 2
    }

    // value_or提供默认值
    auto missing = find_index(v, 99);
    std::cout << missing.value_or(-1) << std::endl;  // -1

    // emplace原地构造
    std::optional<std::string> opt;
    opt.emplace("hello");

    // reset清除
    opt.reset();
    std::cout << opt.has_value() << std::endl;  // 0
}
```

## 三、注意事项

- `*opt`在无值时未定义行为，用`value()`可抛异常
- optional的值存储在内部，无堆分配
- 适用于函数可能失败的场景（比异常轻量）
- C++23的`std::expected`提供更丰富的错误信息
