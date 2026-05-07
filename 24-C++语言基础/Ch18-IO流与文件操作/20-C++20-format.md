# C++20 format

## 一、概念说明

`std::format`（C++20, `<format>`）是现代化的字符串格式化库，受Python的`str.format()`和C的`printf`启发，提供类型安全、可读性好的格式化方式。

语法：`std::format("格式字符串", args...)`，使用`{}`作为占位符。

```cpp
#include <iostream>
#include <format>
#include <string>

int main() {
    // 基本用法
    auto s1 = std::format("Hello, {}!", "World");
    std::cout << s1 << std::endl;

    // 按位置
    auto s2 = std::format("{0} 和 {1}, {0} again", "A", "B");
    std::cout << s2 << std::endl;

    // 类型自动推导
    auto s3 = std::format("整数: {}, 浮点: {}, 布尔: {}", 42, 3.14, true);
    std::cout << s3 << std::endl;

    // 格式说明符
    auto s4 = std::format("{:d}", 255);          // 十进制
    auto s5 = std::format("{:x}", 255);          // 十六进制
    auto s6 = std::format("{:.2f}", 3.14159);    // 两位小数
    auto s7 = std::format("{:>10}", "right");    // 右对齐
    std::cout << s4 << " " << s5 << " " << s6 << " " << s7 << std::endl;

    return 0;
}
```

**输出：**
```
Hello, World!
A 和 B, A again
整数: 42, 浮点: 3.14, 布尔: true
255 ff 3.14     right
```

## 二、具体用法

### 2.1 格式化输出到流

```cpp
#include <iostream>
#include <format>

int main() {
    // print（C++23，部分编译器支持）
    // std::print("Hello, {}!\n", "format");

    // 使用format + cout
    std::cout << std::format("名称: {:<10}, 分数: {:>5}, 等级: {}",
                             "Alice", 95, 'A') << std::endl;

    // 格式化到字符串后输出
    std::string msg = std::format("进度: [{:{}}] {:.0f}%",
                                  std::string(20, '#'), 20, 85.5);
    std::cout << msg << std::endl;

    // 填充和对齐
    std::cout << std::format("{:*^30}", "居中对齐") << std::endl;
    std::cout << std::format("{:.<20}", "左对齐") << std::endl;
    std::cout << std::format("{:.>20}", "右对齐") << std::endl;

    return 0;
}
```

**输出：**
```
名称: Alice      , 分数:    95, 等级: A
进度: [####################] 85%
***********居中对齐************
左对齐............
................右对齐
```

### 2.2 高级格式化

```cpp
#include <iostream>
#include <format>
#include <vector>

// 自定义类型的格式化支持
struct Point {
    int x, y;
};

template <>
struct std::formatter<Point> : std::formatter<std::string> {
    auto format(const Point& p, auto& ctx) const {
        return std::formatter<std::string>::format(
            std::format("({}, {})", p.x, p.y), ctx);
    }
};

int main() {
    // 数字格式化
    std::cout << std::format("{:05d}", 42) << std::endl;       // 00042
    std::cout << std::format("{:+d}", 42) << std::endl;        // +42
    std::cout << std::format("{: d}", -42) << std::endl;       // -42
    std::cout << std::format("{:#x}", 255) << std::endl;       // 0xff

    // 浮点格式化
    std::cout << std::format("{:e}", 1234.5) << std::endl;     // 1.234500e+03
    std::cout << std::format("{:g}", 1234.5) << std::endl;     // 1234.5

    // 自定义类型
    Point p{3, 4};
    std::cout << std::format("点: {}", p) << std::endl;

    return 0;
}
```

**输出：**
```
00042
+42
-42
0xff
1.234500e+03
1234.5
点: (3, 4)
```

## 三、注意事项与常见陷阱

- **需要C++20支持**：GCC 13+、Clang 14+、MSVC 2019 16.10+。
- **`std::format`是类型安全的**：编译期检查参数类型，不像`sprintf`。
- **性能优于`sprintf`和`ostringstream`**：编译期解析格式字符串。
- **`std::print`是C++23特性**：直接输出到流。
- **自定义类型需要特化`std::formatter`**：实现`format`方法。
