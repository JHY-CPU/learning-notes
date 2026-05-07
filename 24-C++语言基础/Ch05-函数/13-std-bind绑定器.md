# std::bind绑定器

## 一、概念说明

`std::bind`是C++11引入的函数适配器，定义在`<functional>`中。它可以将函数的部分参数**预先绑定**，生成一个新的可调用对象。通过**占位符**（`std::placeholders::_1`, `_2`, ...）可以延迟部分参数的传入。

C++11后lambda通常更清晰，但`std::bind`在某些场景仍然有用。

## 二、具体用法

### 2.1 基本绑定

```cpp
#include <functional>
using namespace std::placeholders;

int add(int a, int b) { return a + b; }

// 绑定第一个参数为10
auto add10 = std::bind(add, 10, _1);
std::cout << add10(5) << std::endl;    // 输出: 15
std::cout << add10(20) << std::endl;   // 输出: 30

// 绑定第二个参数
auto addTo100 = std::bind(add, _1, 100);
std::cout << addTo100(7) << std::endl;  // 输出: 107
```

### 2.2 参数重排

```cpp
int divide(int a, int b) { return a / b; }

// 交换参数顺序
auto divideRev = std::bind(divide, _2, _1);
std::cout << divideRev(10, 2) << std::endl;  // 输出: 0 (即 2/10)
```

### 2.3 绑定成员函数

```cpp
class Printer {
public:
    void print(const std::string& msg) {
        std::cout << "Printer: " << msg << std::endl;
    }
};

Printer printer;
auto boundPrint = std::bind(&Printer::print, &printer, _1);
boundPrint("Hello!");
// 输出: Printer: Hello!
```

### 2.4 等价lambda对比

```cpp
// std::bind写法
auto add10_bind = std::bind(add, 10, _1);

// 等价lambda（通常更清晰）
auto add10_lambda = [](int x) { return add(10, x); };
```

## 三、注意事项与常见陷阱

- `std::bind`默认按值拷贝参数，用`std::ref()`可按引用传递
- 占位符从`_1`到`_N`，对应调用时的第1到第N个参数
- `std::bind`的返回类型难以声明，通常用`auto`
- C++11后优先考虑lambda替代`std::bind`，可读性更好
- `std::bind`与`std::function`配合使用实现灵活回调
