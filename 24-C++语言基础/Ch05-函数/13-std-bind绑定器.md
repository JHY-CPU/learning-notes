# std::bind绑定器

## 一、概念说明

`std::bind`是C++11引入的函数适配器（`<functional>`头文件，C++11标准 §20.8.9），可以将函数的部分参数**预先绑定**，生成一个新的可调用对象。通过**占位符**（`std::placeholders::_1`, `_2`, ...）可以延迟部分参数的传入。

与lambda的对比：

| 特性 | std::bind | lambda |
|------|-----------|--------|
| 可读性 | 参数重排时较晦涩 | 直观 |
| 参数重排 | 原生支持 | 需手动编写 |
| 绑定成员函数 | 需显式传this | `[this]`捕获 |
| 性能 | 类型擦除开销 | 编译器内联优化 |
| C++标准推荐 | C++11后逐渐被lambda取代 | 优先使用 |

## 二、具体用法

### 2.1 基本绑定

```cpp
#include <iostream>
#include <functional>
using namespace std;
using namespace std::placeholders;

int add(int a, int b) { return a + b; }

int main() {
    // 绑定第一个参数为10
    auto add10 = bind(add, 10, _1);
    cout << add10(5) << endl;    // 15 (add(10, 5))
    cout << add10(20) << endl;   // 30 (add(10, 20))

    // 绑定第二个参数为100
    auto addTo100 = bind(add, _1, 100);
    cout << addTo100(7) << endl;  // 107 (add(7, 100))

    // 同时绑定两个参数
    auto add5And3 = bind(add, 5, 3);
    cout << add5And3() << endl;   // 8

    return 0;
}
```

### 2.2 参数重排

```cpp
#include <iostream>
#include <functional>
using namespace std;
using namespace std::placeholders;

int divide(int a, int b) { return a / b; }

int main() {
    // 交换参数顺序：调用divideRev(a, b) 实际执行 divide(b, a)
    auto divideRev = bind(divide, _2, _1);
    cout << divideRev(10, 2) << endl;  // 0 (divide(2, 10))

    // 多参数重排
void threeArgs(int a, int b, int c) {
    cout << a << " " << b << " " << c << endl;
}
    auto reorder = bind(threeArgs, _3, _1, _2);
    reorder(1, 2, 3);  // 输出: 3 1 2

    return 0;
}
```

### 2.3 绑定成员函数

```cpp
#include <iostream>
#include <functional>
using namespace std;
using namespace std::placeholders;

class Printer {
public:
    void print(const string& msg) {
        cout << "Printer: " << msg << endl;
    }

    int add(int a, int b) {
        return a + b;
    }
};

int main() {
    Printer printer;

    // 绑定成员函数：第一个参数是对象指针
    auto boundPrint = bind(&Printer::print, &printer, _1);
    boundPrint("Hello!");  // Printer: Hello!

    // 绑定带参数的成员函数
    auto boundAdd = bind(&Printer::add, &printer, _1, _2);
    cout << boundAdd(3, 4) << endl;  // 7

    return 0;
}
```

### 2.4 按引用传递

```cpp
#include <iostream>
#include <functional>
using namespace std;
using namespace std::placeholders;

void increment(int& x) { x++; }

int main() {
    int value = 0;

    // std::bind默认按值拷贝参数
    auto boundInc = bind(increment, value);  // 拷贝value的值
    boundInc();
    cout << value << endl;  // 0（外部value未变）

    // 使用std::ref按引用传递
    auto boundIncRef = bind(increment, ref(value));
    boundIncRef();
    cout << value << endl;  // 1（外部value被修改）

    return 0;
}
```

### 2.5 等价lambda对比

```cpp
#include <iostream>
#include <functional>
using namespace std;
using namespace std::placeholders;

int add(int a, int b) { return a + b; }

int main() {
    // std::bind写法
    auto add10_bind = bind(add, 10, _1);

    // 等价lambda（更清晰）
    auto add10_lambda = [](int x) { return add(10, x); };

    // 复杂bind的lambda等价
    auto complexBind = bind(add, _2, _1);
    // 等价lambda：
    auto complexLambda = [](int a, int b) { return add(b, a); };

    cout << add10_bind(5) << endl;     // 15
    cout << add10_lambda(5) << endl;   // 15

    return 0;
}
```

### 2.6 与std::function配合

```cpp
#include <iostream>
#include <functional>
#include <vector>
using namespace std;
using namespace std::placeholders;

void process(int x, const string& label) {
    cout << label << ": " << x << endl;
}

int main() {
    // 存储不同绑定的回调
    vector<function<void(int)>> callbacks;
    callbacks.push_back(bind(process, _1, "INFO"));
    callbacks.push_back(bind(process, _1, "ERROR"));
    callbacks.push_back(bind(process, _1, "DEBUG"));

    for (auto& cb : callbacks) {
        cb(42);
    }
    // INFO: 42
    // ERROR: 42
    // DEBUG: 42

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **std::bind默认按值拷贝参数**：用`std::ref()`或`std::cref()`可按引用传递
2. **占位符从`_1`到`_N`**：对应调用时的第1到第N个参数
3. **std::bind的返回类型难以声明**：通常用`auto`，实际类型是未指定的闭包类型
4. **C++11后优先考虑lambda替代std::bind**：可读性更好，编译器优化更佳
5. **绑定的参数默认被拷贝**：对不可拷贝类型需用`std::ref`或`std::move`
6. **不要绑定可能被销毁的对象**：bind结果可能长期持有引用/指针
7. **std::bind_front (C++20) 简化了前向绑定**：不需要占位符语法
