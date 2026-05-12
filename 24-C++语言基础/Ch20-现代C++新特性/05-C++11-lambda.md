# C++11 lambda

## 一、概念说明

Lambda是匿名函数对象（C++11 §5.1.2），语法：`[捕获列表](参数) -> 返回类型 { 函数体 }`。编译器为每个lambda生成一个唯一的匿名类，重载`operator()`。Lambda大幅简化了回调和STL算法的使用。

### 1.1 捕获方式

| 捕获 | 含义 |
|------|------|
| `[]` | 不捕获任何变量 |
| `[=]` | 按值捕获所有外部变量 |
| `[&]` | 按引用捕获所有外部变量 |
| `[x]` | 按值捕获x |
| `[&x]` | 按引用捕获x |
| `[this]` | 捕获this指针 |
| `[=, &x]` | 默认按值，x按引用 |
| `[&, x]` | 默认按引用，x按值 |

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    auto greet = []() { std::cout << "Hello Lambda!" << std::endl; };
    greet();

    auto add = [](int a, int b) -> int { return a + b; };
    std::cout << "3+4=" << add(3, 4) << std::endl;

    int factor = 10;
    auto multiply = [factor](int x) { return x * factor; };
    std::cout << "5*10=" << multiply(5) << std::endl;

    // mutable：允许修改按值捕获的副本
    int counter = 0;
    auto increment = [counter]() mutable { return ++counter; };
    std::cout << increment() << " " << increment() << " " << increment() << std::endl;
    std::cout << "原counter=" << counter << std::endl;

    // STL算法中的lambda
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
#include <vector>
#include <string>

int main() {
    // C++14: 泛型lambda（auto参数）
    auto print = [](const auto& val) { std::cout << val << std::endl; };
    print(42);
    print(3.14);
    print("hello");

    // 通用容器求和
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

C++14允许在捕获时创建新变量或移动捕获move-only类型。

```cpp
#include <iostream>
#include <memory>
#include <string>

int main() {
    // 移动unique_ptr到lambda中
    auto ptr = std::make_unique<int>(42);
    auto lambda = [p = std::move(ptr)]() {
        std::cout << "unique_ptr值: " << *p << std::endl;
    };
    lambda();

    // 捕获初始化生成器
    auto gen = [count = 0]() mutable { return ++count; };
    std::cout << gen() << " " << gen() << " " << gen() << std::endl;

    // 捕获字符串字面量
    auto log = [prefix = std::string("[LOG] ")](const std::string& msg) {
        std::cout << prefix << msg << std::endl;
    };
    log("系统启动");
    log("完成初始化");

    return 0;
}
```

**输出：**
```
unique_ptr值: 42
1 2 3
[LOG] 系统启动
[LOG] 完成初始化
```

### 2.3 lambda的常见用途

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 1. 作为STL算法的谓词
    auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 5; });
    if (it != v.end()) std::cout << "第一个>5: " << *it << std::endl;

    // 2. 作为回调存储
    std::vector<std::function<void()>> callbacks;
    for (int i = 0; i < 3; ++i) {
        callbacks.push_back([i]() { std::cout << "回调" << i << std::endl; });
    }
    for (auto& cb : callbacks) cb();

    // 3. 闭包（捕获外部状态）
    int threshold = 3;
    auto count = std::count_if(v.begin(), v.end(),
                               [threshold](int x) { return x > threshold; });
    std::cout << "大于" << threshold << "的元素: " << count << "个" << std::endl;

    return 0;
}
```

**输出：**
```
第一个>5: 9
回调0
回调1
回调2
大于3的元素: 4个
```

## 三、注意事项与常见陷阱

1. **悬垂引用**：lambda引用捕获的变量在lambda调用前已被销毁是最常见的bug。
2. **`[=]`不捕获this的成员**：C++20中`[=]`被弃用，建议用`[this]`或`[=, this]`。
3. **`mutable`只允许修改按值捕获的副本**：不影响原变量。
4. **lambda类型是匿名的**：每个lambda有唯一类型，用`std::function`或`auto`存储。
5. **递归lambda**：需要`std::function`存储，或使用C++14的y-combinator模式。
6. **lambda在类成员中捕获this**：注意对象生命周期，避免this悬垂。
