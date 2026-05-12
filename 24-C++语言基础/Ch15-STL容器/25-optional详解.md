# optional详解

## 一、概念说明

`std::optional`是C++17引入的包装类（C++标准 §20.6.3），表示一个值可能存在也可能不存在。它替代了使用特殊值（如-1、nullptr）表示"无值"的做法，提供类型安全的可选值语义。optional的值存储在内部，无堆分配，性能优于指针方案。

### 1.1 设计动机

```
传统做法的问题：
- 返回-1表示失败：-1可能是有效值
- 返回nullptr：需要指针/堆分配
- 抛异常：性能开销大，不适合正常流程
- 输出参数：调用方容易忘记检查

optional的优势：
- 类型安全：编译期检查
- 零开销：值存储在optional内部
- 自文档化：返回类型明确表达可能失败
- 支持函数式编程风格
```

## 二、具体用法

### 2.1 基本操作

```cpp
#include <optional>
#include <iostream>
#include <string>
#include <vector>

std::optional<int> find_index(const std::vector<int>& vec, int target) {
    for (size_t i = 0; i < vec.size(); ++i)
        if (vec[i] == target) return static_cast<int>(i);
    return std::nullopt;  // 无值
}

int main() {
    std::vector<int> v = {10, 20, 30, 40};

    // 检查值
    auto idx = find_index(v, 30);
    if (idx) {                        // 隐式bool转换
        std::cout << "找到: " << *idx << std::endl;  // 2
    }

    // has_value()
    if (idx.has_value())
        std::cout << idx.value() << std::endl;

    // value_or提供默认值
    auto missing = find_index(v, 99);
    std::cout << missing.value_or(-1) << std::endl;  // -1

    // 创建
    std::optional<std::string> opt1 = std::string("hello");
    auto opt2 = std::make_optional<std::string>("world");
    std::optional<int> opt3;          // 空
    std::optional<int> opt4(42);      // 有值
}
```

### 2.2 修改操作

```cpp
void modify_demo() {
    std::optional<std::string> opt;

    // emplace原地构造
    opt.emplace("hello");

    // 赋值
    opt = "world";

    // reset清除
    opt.reset();
    std::cout << opt.has_value() << std::endl;  // false

    // swap
    std::optional<int> a(1), b(2);
    a.swap(b);  // a=2, b=1
}
```

### 2.3 函数式风格

```cpp
#include <algorithm>

// 链式处理
std::optional<int> parse_int(const std::string& s) {
    try { return std::stoi(s); }
    catch (...) { return std::nullopt; }
}

std::optional<int> double_if_positive(int x) {
    if (x > 0) return x * 2;
    return std::nullopt;
}

void functional_demo() {
    auto result = parse_int("42")
        .and_then(double_if_positive)  // C++23
        .value_or(0);

    // C++17没有and_then，手动链式
    auto opt = parse_int("42");
    if (opt) {
        auto doubled = double_if_positive(*opt);
        if (doubled)
            std::cout << *doubled << std::endl;  // 84
    }
}
```

### 2.4 实用示例

```cpp
// 配置读取
std::optional<std::string> get_env(const std::string& name) {
    const char* val = std::getenv(name.c_str());
    if (val) return std::string(val);
    return std::nullopt;
}

// 缓存查询
std::optional<int> cache_lookup(const std::string& key) {
    static std::map<std::string, int> cache;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return std::nullopt;
}
```

## 三、注意事项与常见陷阱

1. **`*opt`在无值时未定义行为**：用`value()`可抛`bad_optional_access`异常
2. **optional的值存储在内部**：无堆分配，但对象较大时optional也较大
3. **适用于函数可能失败的场景**：比异常轻量，比错误码清晰
4. **C++23的`std::expected`提供更丰富的错误信息**：可同时返回值和错误
5. **`operator->`和`operator*`**：无值时行为未定义，务必先检查
6. **`emplace`直接构造**：避免创建临时对象
7. **optional不支持引用类型**：用`std::reference_wrapper`包装
