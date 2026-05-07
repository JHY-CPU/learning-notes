# constexpr-if与概念

## 一、概念说明

`constexpr if`结合C++20 Concepts（概念），可以编写比传统SFINAE更清晰、更可读的编译期分支。Concepts提供了声明式的约束语义，而`constexpr if`提供运行时逻辑的编译期版本，两者结合是现代C++模板编程的最佳实践。

## 二、具体用法

### 2.1 用concept替代enable_if

```cpp
// 传统SFINAE写法
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::string>
describe_old(T val) {
    return "数字: " + std::to_string(val);
}

// Concept + constexpr if 写法
template <typename T>
    requires std::is_arithmetic_v<T>
std::string describe(T val) {
    if constexpr (std::is_integral_v<T>) {
        return "整数: " + std::to_string(val);
    } else {
        return "浮点: " + std::to_string(val);
    }
}

int main() {
    std::cout << describe(42) << std::endl;    // 整数: 42
    std::cout << describe(3.14) << std::endl;  // 浮点: 3.140000
}
```

### 2.2 编译期策略选择

```cpp
// 不同存储策略
template <typename T>
void optimized_process(T value) {
    if constexpr (sizeof(T) <= 8) {
        // 小类型：按值传递，直接操作
        std::cout << "栈上处理: " << value << std::endl;
    } else if constexpr (std::is_move_constructible_v<T>) {
        // 大类型但可移动：移动语义
        T moved = std::move(value);
        std::cout << "移动处理" << std::endl;
    } else {
        // 不可移动：复制
        T copy = value;
        std::cout << "复制处理" << std::endl;
    }
}

int main() {
    optimized_process(42);                       // 栈上处理: 42
    optimized_process(std::string("hello"));     // 移动处理
    optimized_process(std::vector<int>(1000));   // 移动处理
}
```

### 2.3 requires表达式增强条件

```cpp
// 检测是否有特定成员函数
template <typename T>
void smart_serialize(const T& obj) {
    if constexpr (requires { obj.serialize(); }) {
        // 类型有serialize()方法
        std::cout << "自定义序列化: " << obj.serialize() << std::endl;
    } else if constexpr (requires { std::to_string(obj); }) {
        // 可转为字符串
        std::cout << "to_string: " << std::to_string(obj) << std::endl;
    } else {
        std::cout << "无序列化支持" << std::endl;
    }
}

struct Config {
    std::string name;
    std::string serialize() const { return "Config{" + name + "}"; }
};

int main() {
    Config cfg{"test"};
    smart_serialize(cfg);      // 自定义序列化: Config{test}
    smart_serialize(42);       // to_string: 42
    smart_serialize(std::vector<int>{}); // 无序列化支持
}
```

### 2.4 编译期类型分发

```cpp
template <typename T>
auto get_parser() {
    if constexpr (std::is_same_v<T, int>) {
        return [](const std::string& s) { return std::stoi(s); };
    } else if constexpr (std::is_same_v<T, double>) {
        return [](const std::string& s) { return std::stod(s); };
    } else if constexpr (std::is_same_v<T, std::string>) {
        return [](const std::string& s) { return s; };
    }
}

int main() {
    auto int_parser = get_parser<int>();
    auto dbl_parser = get_parser<double>();
    std::cout << int_parser("42") << std::endl;    // 42
    std::cout << dbl_parser("3.14") << std::endl;  // 3.14
}
```

## 三、注意事项与常见陷阱

- 每个分支只需语法正确，不要求类型完全匹配（未选中分支不实例化）
- `requires`表达式可以嵌套在`if constexpr`条件中
- Concept约束放在模板参数列表中，`if constexpr`放在函数体内
- 编译器可能对每个分支生成不同的警告，可用`if constexpr`抑制无关警告
- 与传统SFINAE相比，错误信息更清晰，调试更容易
- 注意`if constexpr`不能用于非模板函数的编译期分支
