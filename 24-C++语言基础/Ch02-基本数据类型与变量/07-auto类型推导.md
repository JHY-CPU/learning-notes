# auto类型推导

## 一、概念说明

`auto`关键字让编译器根据初始化表达式**自动推导变量类型**。C++11引入的`auto`大幅简化了复杂类型声明，特别在使用模板和迭代器时。

## 二、具体用法

### 2.1 auto基本用法

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    // 编译器推导类型
    auto x = 42;              // int
    auto y = 3.14;            // double
    auto z = 'A';             // char
    auto name = string("张三"); // string
    auto flag = true;         // bool

    cout << "x类型大小: " << sizeof(x) << endl;
    cout << "y类型大小: " << sizeof(y) << endl;

    // 使用auto简化复杂类型
    vector<int> nums = {1, 2, 3, 4, 5};

    // 不用auto：繁琐
    for (vector<int>::iterator it = nums.begin(); it != nums.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // 用auto：简洁
    for (auto it = nums.begin(); it != nums.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    return 0;
}
```

输出：
```
x类型大小: 4
y类型大小: 8
1 2 3 4 5
1 2 3 4 5
```

### 2.2 auto推导规则

```cpp
#include <iostream>
#include <typeinfo>
using namespace std;

int main() {
    // 规则1：auto丢弃引用
    int a = 10;
    int& ref = a;
    auto b = ref;        // b是int，不是int&

    // 规则2：auto保留const（但丢弃引用的const）
    const int c = 42;
    auto d = c;          // d是int（const被丢弃，因为c是顶层const）
    const auto e = c;    // e是const int

    // 规则3：auto&保留引用和const
    auto& f = ref;       // f是int&
    const auto& g = c;   // g是const int&

    // 规则4：auto推导数组为指针
    int arr[] = {1, 2, 3};
    auto ptr = arr;      // ptr是int*（数组退化为指针）

    // 验证
    b = 20;
    cout << "a = " << a << ", b = " << b << endl;  // a不受影响

    d = 30;
    // e = 30;  // 编译错误：e是const

    return 0;
}
```

输出：
```
a = 10, b = 20
```

### 2.3 auto与引用、指针

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<string> names = {"Alice", "Bob", "Charlie"};

    // auto：拷贝元素
    for (auto name : names) {
        name = "Modified";  // 不影响原vector
    }
    cout << "拷贝后: " << names[0] << endl;

    // auto&：引用元素，可以修改
    for (auto& name : names) {
        name = "Modified";
    }
    cout << "引用后: " << names[0] << endl;

    // const auto&：只读引用，避免拷贝
    vector<string> items = {"A", "B", "C"};
    for (const auto& item : items) {
        cout << item << " ";
    }
    cout << endl;

    return 0;
}
```

输出：
```
拷贝后: Alice
引用后: Modified
A B C
```

### 2.4 auto的限制

```cpp
#include <iostream>
#include <vector>
using namespace std;

// 错误：函数参数不能用auto（C++14前）
// void func(auto x) {}  // C++14前不合法

// C++14允许lambda参数用auto
auto lambda = [](auto a, auto b) { return a + b; };

// auto不能推导的情况
int main() {
    // auto x;             // 错误：必须有初始化
    // auto arr[] = {1,2}; // 错误：不能推导数组类型

    // 正确用法
    auto x = 42;
    auto arr = {1, 2, 3}; // 推导为initializer_list<int>

    cout << "lambda(1, 2) = " << lambda(1, 2) << endl;
    cout << "lambda(1.5, 2.5) = " << lambda(1.5, 2.5) << endl;

    return 0;
}
```

输出：
```
lambda(1, 2) = 3
lambda(1.5, 2.5) = 4
```

## 三、注意事项与常见陷阱

1. **auto必须初始化**：`auto x;`是错误的，编译器无法推导类型
2. **auto丢弃引用**：`auto b = ref;`中b是值类型而非引用，修改b不影响原变量
3. **auto不是动态类型**：auto在编译时确定类型，不是运行时类型推导
4. **可读性权衡**：类型不明显时（如函数返回值），使用auto可能降低可读性
5. **auto&用于范围for**：遍历容器并修改元素时，必须使用`auto&`或`const auto&`
