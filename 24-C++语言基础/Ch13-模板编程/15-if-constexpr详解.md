# if constexpr详解

## 一、概念说明

`if constexpr`是C++17引入的编译期条件语句。与普通`if`不同，`if constexpr`在编译时求值条件，不满足分支的代码不会被实例化。这大大简化了模板中的分支逻辑，避免了SFINAE的复杂写法。

## 二、具体用法

### 2.1 基本用法

```cpp
template <typename T>
std::string type_to_string(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        return "整数: " + std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "浮点: " + std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "字符串: " + value;
    } else {
        return "未知类型";
    }
}

int main() {
    std::cout << type_to_string(42) << std::endl;           // 整数: 42
    std::cout << type_to_string(3.14) << std::endl;         // 浮点: 3.140000
    std::cout << type_to_string(std::string("hi")) << std::endl; // 字符串: hi
}
```

### 2.2 替代SFINAE

```cpp
// 使用SFINAE的旧写法
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, T>
square_old(T x) { return x * x; }

// 使用if constexpr的新写法（更清晰）
template <typename T>
auto square(T x) {
    if constexpr (std::is_arithmetic_v<T>) {
        return x * x;
    } else if constexpr (std::is_same_v<T, std::string>) {
        return x + x;  // 字符串"平方"：重复拼接
    }
}

int main() {
    std::cout << square(5) << std::endl;                    // 25
    std::cout << square(2.5) << std::endl;                  // 6.25
    std::cout << square(std::string("ab")) << std::endl;    // abab
}
```

### 2.3 编译期递归终止

```cpp
// 打印变参模板的所有参数
template <typename T, typename... Args>
void print_all(T first, Args... rest) {
    std::cout << first;
    if constexpr (sizeof...(rest) > 0) {
        std::cout << ", ";
        print_all(rest...);  // 仅在有剩余参数时递归
    } else {
        std::cout << std::endl;
    }
}

int main() {
    print_all(1, 2.5, "hello", 'A');  // 1, 2.5, hello, A
    print_all(42);                     // 42
}
```

### 2.4 配合concept使用

```cpp
template <typename T>
void smart_print(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "[int] " << value << std::endl;
    } else if constexpr (requires { std::cout << value; }) {
        // requires表达式（C++20 concept语法）
        std::cout << value << std::endl;
    } else {
        std::cout << "[不可打印]" << std::endl;
    }
}

int main() {
    smart_print(42);         // [int] 42
    smart_print("hello");    // hello
    smart_print(std::vector<int>{1, 2});  // [不可打印]
}
```

## 三、注意事项与常见陷阱

- `if constexpr`的条件必须是编译期常量表达式
- 不满足分支的代码不会被实例化，但仍需语法正确
- 不能在非模板函数中使用`if constexpr`（没有模板参数可依赖）
- `if constexpr`不能用于运行时变量
- 每个分支可以返回不同类型（普通if不能）
- `else if constexpr`可以链式使用，但注意最后一个应为`else`
- 在C++20中可结合`requires`表达式使条件更灵活
