# Lambda表达式进阶

## 一、概念说明

C++14及之后对lambda表达式进行了多项增强：泛型lambda（auto参数）、`mutable`关键字、`constexpr` lambda（C++17）、捕获初始化（C++14）等。这些特性使lambda更加强大和灵活。

## 二、具体用法

### 2.1 泛型lambda（C++14）

```cpp
// auto参数使lambda成为模板
auto print = [](const auto& value) {
    std::cout << value << std::endl;
};

print(42);          // 输出: 42
print(3.14);        // 输出: 3.14
print("hello");     // 输出: hello

// 泛型lambda与STL
auto greater = [](const auto& a, const auto& b) {
    return a > b;
};
std::cout << greater(10, 5) << std::endl;     // 输出: 1
std::cout << greater(1.5, 2.5) << std::endl;  // 输出: 0
```

### 2.2 mutable关键字

```cpp
int x = 10;
auto lambda = [x]() mutable {
    x++;  // 不加mutable会编译错误（值捕获默认const）
    std::cout << "内部 x = " << x << std::endl;
};
lambda();
lambda();
std::cout << "外部 x = " << x << std::endl;
// 输出:
// 内部 x = 11
// 内部 x = 12
// 外部 x = 10    // 原变量不变
```

### 2.3 捕获初始化（C++14）

```cpp
// 在捕获列表中创建新变量
auto counter = [count = 0]() mutable { return ++count; };
std::cout << counter() << std::endl;  // 输出: 1
std::cout << counter() << std::endl;  // 输出: 2

// 移动捕获（unique_ptr等只移类型）
auto ptr = std::make_unique<int>(42);
auto lambda = [p = std::move(ptr)]() {
    std::cout << *p << std::endl;
};
lambda();
// 输出: 42
```

### 2.4 constexpr lambda（C++17）

```cpp
// C++17起lambda默认在满足条件时为constexpr
auto square = [](int x) { return x * x; };

constexpr int result = square(5);  // 编译期求值
static_assert(result == 25, "should be 25");
```

## 三、注意事项与常见陷阱

- `mutable`只影响值捕获的副本，不影响原变量
- 泛型lambda的`auto`参数本质上是模板参数
- 捕获初始化中`=`左侧是lambda内部名，右侧是初始化表达式
- 移动捕获是C++14的重要特性，解决了unique_ptr等不可拷贝类型的捕获问题
- `constexpr` lambda要求函数体满足constexpr的约束条件
