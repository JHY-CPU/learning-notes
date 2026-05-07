# 替代方案：Result类型

## 一、概念说明

Result类型（如`std::expected`、`std::optional`、`std::variant`）提供了一种不使用异常的错误处理方式。它们将错误作为返回值的一部分，适合高频错误和性能敏感场景。

## 二、具体用法

### 2.1 std::optional（C++17）

```cpp
#include <iostream>
#include <optional>
#include <string>

// optional表示可能无值
std::optional<int> safe_divide(int a, int b) {
    if (b == 0) return std::nullopt;
    return a / b;
}

std::optional<std::string> find_user(int id) {
    if (id == 1) return "张三";
    if (id == 2) return "李四";
    return std::nullopt;
}

int main() {
    auto result = safe_divide(10, 3);
    if (result) {
        std::cout << "结果: " << *result << std::endl;  // 3
    } else {
        std::cout << "除零错误" << std::endl;
    }

    auto name = find_user(1);
    std::cout << name.value_or("未知用户") << std::endl;  // 张三
}
```

### 2.2 std::variant（C++17）

```cpp
#include <variant>

// variant：类型安全的union，可存储结果或错误
template <typename T, typename E = std::string>
using Result = std::variant<T, E>;

Result<int, std::string> parse_int(const std::string& s) {
    try {
        return std::stoi(s);
    } catch (...) {
        return "解析失败: " + s;
    }
}

int main() {
    auto result = parse_int("42");

    // 使用visit访问
    std::visit([](auto&& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, int>) {
            std::cout << "成功: " << val << std::endl;
        } else {
            std::cout << "错误: " << val << std::endl;
        }
    }, result);

    // 使用holds_alternative检查
    if (std::holds_alternative<int>(result)) {
        std::cout << "值: " << std::get<int>(result) << std::endl;
    }
}
```

### 2.3 std::expected（C++23）

```cpp
// C++23 std::expected — 最佳的Result类型
// template <typename T, typename E>
// class expected;

// 语法示例（需要C++23支持）
/*
std::expected<int, std::string> divide(int a, int b) {
    if (b == 0) return std::unexpected("除数为零");
    return a / b;
}

int main() {
    auto result = divide(10, 2);
    if (result) {
        std::cout << *result << std::endl;
    } else {
        std::cout << result.error() << std::endl;
    }
}
*/
```

### 2.4 异常 vs Result类型

```cpp
/*
| 特性          | 异常                 | Result类型           |
|--------------|---------------------|---------------------|
| 性能（无错误） | 零开销              | 微小开销             |
| 性能（有错误） | 高开销（栈展开）     | 低开销（返回值）      |
| 强制处理      | 是（不捕获则终止）   | 否（可忽略）         |
| 传播方式      | 自动沿栈传播         | 需手动检查和传播      |
| 代码侵入      | 低                  | 高                   |
| 适用场景      | 罕见错误、跨层传播   | 高频错误、性能关键    |
*/
```

### 2.5 实用的Result封装

```cpp
// 简化的Result实现
template <typename T, typename E = std::string>
class Result {
    std::variant<T, E> data;
public:
    static Result ok(T value) { return Result(std::move(value)); }
    static Result err(E error) { return Result(std::move(error)); }

    bool is_ok() const { return std::holds_alternative<T>(data); }
    T& value() { return std::get<T>(data); }
    E& error() { return std::get<E>(data); }

    // map操作
    template <typename F>
    auto map(F&& f) -> Result<decltype(f(std::declval<T>())), E> {
        if (is_ok()) return Result<decltype(f(std::declval<T>())), E>::ok(f(value()));
        return Result<decltype(f(std::declval<T>())), E>::err(error());
    }

private:
    Result(T v) : data(std::move(v)) {}
    Result(E e) : data(std::move(e)) {}
};

// 使用
auto parse_and_double(const std::string& s) {
    return Result<int>::ok(42).map([](int v) { return v * 2; });
}
```

## 三、注意事项与常见陷阱

- `std::optional`适合简单的存在/不存在情况
- `std::variant`可存储任意类型但访问较繁琐
- `std::expected`（C++23）是最理想的Result类型
- Result类型不能自动传播（需要手动检查）
- 组合多个Result操作时可能产生嵌套检查
- 在性能关键路径上优先使用Result而非异常
