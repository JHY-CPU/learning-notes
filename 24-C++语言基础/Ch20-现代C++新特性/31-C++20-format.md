# C++20 format

## 一、概念说明

`std::format`（`<format>`）是类型安全的字符串格式化库，语法类似Python的`str.format()`。替代不安全的`sprintf`和笨重的`stringstream`。

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
| 对齐 | `<`, `>`, `^` | 左/右/居中 |
| 宽度 | `10` | 最小宽度 |
| 精度 | `.2` | 小数位/字符串长度 |
| 类型 | `d`, `f`, `x`, `b` | 输出格式 |

## 三、注意事项与常见陷阱

- **类型安全**：编译期检查，不像`sprintf`。
- **需要C++20**：GCC 13+、Clang 14+、MSVC 19.29+。
- **`std::print`是C++23特性**：直接输出。
- **自定义类型需要特化`std::formatter`**。
- **详细内容参见Ch18 format章节**。
