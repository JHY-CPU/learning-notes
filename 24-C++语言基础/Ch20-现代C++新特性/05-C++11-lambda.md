# C++11 lambda

## 一、概念说明

Lambda是匿名函数对象，语法：`[捕获列表](参数) -> 返回类型 { 函数体 }`。它是函数对象的语法糖，编译器生成一个匿名类。

捕获方式：
- `[]`：不捕获
- `[=]`：按值捕获所有
- `[&]`：按引用捕获所有
- `[x]`：按值捕获x
- `[&x]`：按引用捕获x
- `[this]`：捕获this指针
- `[=, &x]`：默认按值，x按引用

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    // 基本lambda
    auto greet = []() { std::cout << "Hello Lambda!" << std::endl; };
    greet();

    // 带参数和返回类型
    auto add = [](int a, int b) -> int { return a + b; };
    std::cout << "3+4=" << add(3, 4) << std::endl;

    // 捕获外部变量
    int factor = 10;
    auto multiply = [factor](int x) { return x * factor; };
    std::cout << "5*10=" << multiply(5) << std::endl;

    // mutable：允许修改按值捕获的变量
    int counter = 0;
    auto increment = [counter]() mutable { return ++counter; };
    std::cout << increment() << " " << increment() << " " << increment() << std::endl;
    std::cout << "原counter=" << counter << std::endl;

    // 在STL算法中使用
    std::vector<int> nums = {5, 3, 1, 4, 2};
    std::sort(nums.begin(), nums.end(), [](int a, int b) { return a > b; });
    for (int n : nums) std::cout << n << " ";
    std::cout << std::endl;

    return 0;
}
```

**输出：**
```
Hello Lambda!
3+4=7
5*10=50
1 2 3
原counter=0
5 4 3 2 1
```

## 二、具体用法

### 2.1 泛型lambda（C++14）

```cpp
#include <iostream>

int main() {
    // C++14: 泛型lambda（auto参数）
    auto print = [](const auto& val) { std::cout << val << std::endl; };
    print(42);
    print(3.14);
    print("hello");

    // 泛型lambda用于容器
    auto sum = [](const auto& container) {
        typename std::decay_t<decltype(container)>::value_type total{};
        for (const auto& item : container) total += item;
        return total;
    };

    std::vector<int> vi = {1, 2, 3};
    std::vector<double> vd = {1.1, 2.2, 3.3};
    std::cout << "int sum=" << sum(vi) << std::endl;
    std::cout << "double sum=" << sum(vd) << std::endl;

    return 0;
}
```

**输出：**
```
42
3.14
hello
int sum=6
double sum=6.6
```

### 2.2 捕获初始化（C++14）

```cpp
#include <iostream>
#include <memory>

int main() {
    // C++14: 捕获时初始化（move-only类型）
    auto ptr = std::make_unique<int>(42);
    auto lambda = [p = std::move(ptr)]() {
        std::cout << "unique_ptr值: " << *p << std::endl;
    };
    lambda();

    // 捕获表达式
    auto gen = [count = 0]() mutable { return ++count; };
    std::cout << gen() << " " << gen() << " " << gen() << std::endl;

    return 0;
}
```

**输出：**
```
unique_ptr值: 42
1 2 3
```

## 三、注意事项与常见陷阱

- **悬垂引用**：lambda引用捕获的变量在lambda调用前已被销毁。
- **`[=]`不捕获this的成员**：需要`[this]`或`[=, this]`（C++20）。
- **`mutable`只允许修改按值捕获的副本**：不影响原变量。
- **lambda类型是匿名的**：每个lambda有唯一类型，用`std::function`或`auto`存储。
- **递归lambda**：用`std::function`存储，或C++14的`y-combinator`模式。
