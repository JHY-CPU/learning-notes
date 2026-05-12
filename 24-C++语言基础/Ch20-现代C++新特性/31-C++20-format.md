# C++20 format

## 一、概念说明

`std::format`（C++20 §20.20，`<format>`）是类型安全的字符串格式化库，语法类似Python的`str.format()`。替代不安全的`sprintf`和笨重的`stringstream`，结合了两者优点：简洁的语法和类型安全。

### 1.1 三者对比

| 特性 | sprintf | stringstream | std::format |
|------|---------|-------------|-------------|
| 类型安全 | 否 | 是 | 是 |
| 性能 | 高 | 低 | 高 |
| 语法 | C风格 | 流操作 | 模板字符串 |
| 内存安全 | 否 | 是 | 是 |

```cpp
#include <iostream>
#include <format>
#include <string>

int main() {
    // 基本格式化
    auto s1 = std::format("Hello, {}!", "World");
    auto s2 = std::format("{} + {} = {}", 3, 4, 3 + 4);
    std::cout << s1 << std::endl;
    std::cout << s2 << std::endl;

    // 位置参数
    auto s3 = std::format("{0} 和 {1}, {0} again", "A", "B");
    std::cout << s3 << std::endl;

    // 格式说明符
    auto s4 = std::format("{:.2f}", 3.14159);
    auto s5 = std::format("{:05d}", 42);
    auto s6 = std::format("{:>10}", "right");
    auto s7 = std::format("{:*^20}", "center");
    std::cout << s4 << " " << s5 << " " << s6 << " " << s7 << std::endl;

    return 0;
}
```

**输出：**
```
Hello, World!
3 + 4 = 7
A 和 B, A again
3.14 00042      right *******center*******
```

## 二、具体用法

### 2.1 格式说明符语法

```
{[位置]:[填充][对齐][宽度][.精度][类型]}
```

| 组件 | 示例 | 含义 |
|------|------|------|
| 位置 | `{0}`, `{1}` | 参数索引 |
| 填充 | `*`, `0` | 填充字符 |
| 对齐 | `<`, `>`, `^` | 左对齐/右对齐/居中 |
| 宽度 | `10` | 最小宽度 |
| 精度 | `.2` | 小数位/字符串长度 |
| 类型 | `d`, `f`, `x`, `b` | 输出格式 |

### 2.2 各种格式

```cpp
#include <iostream>
#include <format>

int main() {
    // 整数格式
    std::cout << std::format("十进制: {:d}", 255) << std::endl;
    std::cout << std::format("十六进制: {:x}", 255) << std::endl;
    std::cout << std::format("八进制: {:o}", 255) << std::endl;
    std::cout << std::format("二进制: {:b}", 255) << std::endl;
    std::cout << std::format("带前缀: {:#x}", 255) << std::endl;

    // 浮点格式
    std::cout << std::format("固定: {:.2f}", 3.14159) << std::endl;
    std::cout << std::format("科学: {:.2e}", 12345.678) << std::endl;
    std::cout << std::format("通用: {:g}", 12345.678) << std::endl;

    // 对齐和填充
    std::cout << std::format("左对齐: {:<10}|", "hi") << std::endl;
    std::cout << std::format("右对齐: {:>10}|", "hi") << std::endl;
    std::cout << std::format("居中: {:*^10}|", "hi") << std::endl;
    std::cout << std::format("零填充: {:08d}", 42) << std::endl;

    return 0;
}
```

**输出：**
```
十进制: 255
十六进制: ff
八进制: 377
二进制: 11111111
带前缀: 0xff
固定: 3.14
科学: 1.23e+04
通用: 12345.7
左对齐: hi        |
右对齐:         hi|
居中: ****hi****|
零填充: 00000042
```

### 2.3 自定义类型格式化

```cpp
#include <iostream>
#include <format>

struct Point {
    double x, y;
};

// 特化std::formatter
template <>
struct std::formatter<Point> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const Point& p, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "({}, {})", p.x, p.y);
    }
};

int main() {
    Point p{3.0, 4.0};
    std::cout << std::format("Point: {}", p) << std::endl;
    return 0;
}
```

**输出：**
```
Point: (3, 4)
```

## 三、注意事项与常见陷阱

1. **类型安全**：编译期检查参数类型，不像`sprintf`可能导致未定义行为。
2. **需要C++20**：GCC 13+、Clang 14+、MSVC 19.29+完整支持。
3. **`std::print`是C++23特性**：直接输出到stdout，不需`std::cout <<`。
4. **自定义类型需要特化`std::formatter`**：实现`parse`和`format`方法。
5. **格式字符串是编译期检查的**（C++26）：C++20是运行时解析。
6. **性能优于`stringstream`**：接近`sprintf`，但更安全。
7. **详细内容参见Ch18 IO流章节**。
