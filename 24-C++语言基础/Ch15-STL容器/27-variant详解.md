# variant详解

## 一、概念说明

`std::variant`是C++17引入的类型安全联合体（C++标准 §20.7.2），可以在编译时确定可能的类型集合。相比`any`，它提供更高效的访问（无堆分配）和编译时类型检查。variant可以看作"编译时确定类型的any"。

### 1.1 variant vs union vs any

| 特性 | variant | union | any |
|------|---------|-------|-----|
| 类型安全 | 是 | 否 | 是 |
| 访问方式 | get/visit | 直接成员 | any_cast |
| 非平凡类型 | 支持 | 不支持 | 支持 |
| 性能 | 最高 | 最高 | 中 |
| 类型集合 | 编译时确定 | 编译时确定 | 运行时任意 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <variant>
#include <iostream>
#include <string>

int main() {
    // 存储不同类型（编译时确定）
    std::variant<int, double, std::string> v = 42;

    // 检查当前类型索引
    std::cout << "index: " << v.index() << std::endl;  // 0 (int)

    // 访问
    std::cout << std::get<int>(v) << std::endl;  // 42

    // 修改（自动销毁旧值，构造新值）
    v = 3.14;
    std::cout << "index: " << v.index() << std::endl;  // 1 (double)
    std::cout << std::get<double>(v) << std::endl;  // 3.14

    v = std::string("hello");
    std::cout << "index: " << v.index() << std::endl;  // 2 (string)

    // 安全访问（指针版本，推荐）
    auto* p = std::get_if<int>(&v);
    if (p) std::cout << "是int: " << *p << std::endl;
    else std::cout << "不是int" << std::endl;
}
```

### 2.2 visit访问

```cpp
#include <variant>
#include <iostream>

void visit_demo() {
    std::variant<int, double, std::string> v = 42;

    // visit：处理所有可能类型
    std::visit([](auto&& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, int>)
            std::cout << "int: " << val << std::endl;
        else if constexpr (std::is_same_v<T, double>)
            std::cout << "double: " << val << std::endl;
        else if constexpr (std::is_same_v<T, std::string>)
            std::cout << "string: " << val << std::endl;
    }, v);

    // 重载模式（C++17常用技巧）
    template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
    template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

    v = std::string("hello");
    std::visit(overloaded{
        [](int i) { std::cout << "int: " << i << std::endl; },
        [](double d) { std::cout << "double: " << d << std::endl; },
        [](const std::string& s) { std::cout << "string: " << s << std::endl; }
    }, v);
}
```

### 2.3 monostate处理默认构造

```cpp
#include <variant>

// variant的第一个类型必须可默认构造
// 如果都不行，用std::monostate作为占位
struct NoDefault {
    NoDefault(int) {}  // 无默认构造
};

std::variant<std::monostate, NoDefault, std::string> v;  // 默认是monostate
// v.index() == 0，持有monostate
```

### 2.4 实用示例

```cpp
// 类型安全的错误处理
template <typename T, typename E = std::string>
using Result = std::variant<T, E>;

Result<int> divide(int a, int b) {
    if (b == 0) return std::string("除零错误");
    return a / b;
}

void result_demo() {
    auto r = divide(10, 3);
    std::visit(overloaded{
        [](int val) { std::cout << "结果: " << val << std::endl; },
        [](const std::string& err) { std::cout << "错误: " << err << std::endl; }
    }, r);
}

// 状态机
using State = std::variant<
    struct Idle {},
    struct Running { int progress; },
    struct Done { std::string result; },
    struct Error { std::string message; }
>;
```

## 三、注意事项与常见陷阱

1. **variant的大小是最大类型的大小+类型索引**：`sizeof`可能较大
2. **`std::get<T>`类型不匹配时抛出`bad_variant_access`**
3. **`std::get_if<T>`返回指针**：失败返回nullptr，推荐使用
4. **visit需要处理所有可能类型**：用`overloaded`模式简化
5. **variant不可持有引用类型**：但可持有`reference_wrapper`
6. **`monostate`作为占位**：当第一个类型不可默认构造时
7. **variant是值类型**：赋值时销毁旧值构造新值
