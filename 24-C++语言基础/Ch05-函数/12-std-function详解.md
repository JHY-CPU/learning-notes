# std::function详解

## 一、概念说明

`std::function`是C++11引入的**通用多态函数包装器**，定义在`<functional>`头文件中。它可以存储、复制和调用任何可调用对象——普通函数、lambda、仿函数、绑定表达式等。

`std::function`提供了类型擦除能力，使得不同类型的可调用对象可以统一存储。

## 二、具体用法

### 2.1 包装不同类型的可调用对象

```cpp
#include <functional>

int add(int a, int b) { return a + b; }

struct Multiply {
    int operator()(int a, int b) const { return a * b; }
};

int main() {
    std::function<int(int, int)> op;

    // 包装普通函数
    op = add;
    std::cout << op(3, 4) << std::endl;    // 输出: 7

    // 包装lambda
    op = [](int a, int b) { return a - b; };
    std::cout << op(10, 4) << std::endl;   // 输出: 6

    // 包装仿函数
    op = Multiply{};
    std::cout << op(3, 5) << std::endl;    // 输出: 15
}
```

### 2.2 作为回调类型

```cpp
class Button {
    std::function<void()> onClick;
public:
    void setOnClick(std::function<void()> callback) {
        onClick = std::move(callback);
    }
    void click() {
        if (onClick) onClick();
    }
};

Button btn;
int clickCount = 0;
btn.setOnClick([&clickCount]() {
    clickCount++;
    std::cout << "Clicked " << clickCount << " times\n";
});
btn.click();
btn.click();
// 输出:
// Clicked 1 times
// Clicked 2 times
```

### 2.3 检查是否为空

```cpp
std::function<void()> f;
if (!f) std::cout << "f is empty\n";
// 输出: f is empty

f = []() { std::cout << "hello\n"; };
if (f) f();
// 输出: hello
```

## 三、注意事项与常见陷阱

- `std::function`有**性能开销**（堆分配、虚函数调用），热路径慎用
- 存储的可调用对象必须与签名**类型兼容**
- `std::function`的拷贝要求存储的对象也可拷贝
- 空`std::function`调用抛出`std::bad_function_call`异常
- 对于简单回调，lambda或函数指针可能比`std::function`更高效
