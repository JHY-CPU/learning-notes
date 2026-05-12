# Lambda表达式基础

## 一、概念说明

Lambda表达式是C++11引入的**匿名函数**机制（C++11标准 §5.1.2），允许在表达式位置直接定义函数对象（闭包类型）。它由捕获列表、参数列表、返回类型和函数体组成，极大简化了回调和短小函数的写法。

基本语法：`[捕获列表](参数列表) mutable 异常属性 -> 返回类型 { 函数体 }`

Lambda的实质是编译器生成的匿名类（闭包类型），每个lambda表达式生成唯一的类型。

## 二、具体用法

### 2.1 基本lambda

```cpp
#include <iostream>
#include <functional>
using namespace std;

int main() {
    // 最简单的lambda
    auto hello = []() { cout << "Hello Lambda!" << endl; };
    hello();  // Hello Lambda!

    // 带参数的lambda
    auto add = [](int a, int b) { return a + b; };
    cout << "3 + 5 = " << add(3, 5) << endl;  // 8

    // 返回类型可省略（编译器自动推导）
    auto divide = [](double a, double b) { return a / b; };

    // 显式返回类型
    auto safeDivide = [](double a, double b) -> optional<double> {
        if (b == 0) return nullopt;
        return a / b;
    };

    return 0;
}
```

### 2.2 捕获列表详解

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    int x = 10;
    string name = "Alice";

    // [x] — 值捕获：拷贝x到lambda中（默认const）
    auto byVal = [x](int n) { return n + x; };
    cout << byVal(5) << endl;  // 15

    // [&x] — 引用捕获：直接引用外部变量
    int counter = 0;
    auto increment = [&counter]() { counter++; };
    increment();
    increment();
    cout << "counter = " << counter << endl;  // 2

    // [=] — 捕获所有外部变量（值捕获）
    auto allByVal = [=]() { return x + name.size(); };

    // [&] — 捕获所有外部变量（引用捕获）
    auto allByRef = [&]() { x++; name += "!"; };

    // [=, &name] — 默认值捕获，name引用捕获
    auto mixed1 = [=, &name]() { name += "!"; return x; };

    // [&, x] — 默认引用捕获，x值捕获
    auto mixed2 = [&, x]() { name += "!"; return x; };

    return 0;
}
```

### 2.3 mutable关键字

```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 10;

    // 值捕获的变量默认const，不能修改
    auto bad = [x]() {
        // x = 20;  // 编译错误：x是const
        return x;
    };

    // 使用mutable允许修改值捕获的副本
    auto mutableLambda = [x]() mutable {
        x = 20;  // OK：修改的是lambda内部的副本
        return x;
    };

    cout << mutableLambda() << endl;  // 20
    cout << x << endl;                // 10（外部x未改变）

    // 注意：引用捕获的修改会影响外部
    int y = 10;
    auto refLambda = [&y]() { y = 30; };
    refLambda();
    cout << y << endl;  // 30（外部y被修改）

    return 0;
}
```

### 2.4 在STL算法中使用

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> nums = {5, 2, 8, 1, 9, 3};

    // 排序（降序）
    sort(nums.begin(), nums.end(), [](int a, int b) { return a > b; });

    // 过滤：查找第一个大于5的元素
    auto it = find_if(nums.begin(), nums.end(), [](int x) { return x > 5; });

    // 变换：每个元素乘以2
    transform(nums.begin(), nums.end(), nums.begin(), [](int x) { return x * 2; });

    // 累加：C++17 reduce
    int sum = 0;
    for_each(nums.begin(), nums.end(), [&sum](int x) { sum += x; });

    // 统计满足条件的元素个数
    int count = count_if(nums.begin(), nums.end(), [](int x) { return x > 10; });

    cout << "sum=" << sum << " count=" << count << endl;

    return 0;
}
```

### 2.5 lambda与函数指针

```cpp
// 无捕获的lambda可转换为函数指针
using FuncPtr = int(*)(int, int);

FuncPtr addPtr = [](int a, int b) { return a + b; };  // OK
cout << addPtr(3, 4) << endl;  // 7

// 有捕获的lambda不能转换为函数指针
int offset = 10;
// FuncPtr bad = [offset](int a, int b) { return a + b + offset; };  // 编译错误
```

### 2.6 C++14泛型lambda

```cpp
// C++14：auto参数的lambda（泛型lambda）
auto generic = [](auto a, auto b) { return a + b; };

cout << generic(1, 2) << endl;        // int版本，3
cout << generic(1.5, 2.5) << endl;    // double版本，4.0
cout << generic("hi"s, " there"s) << endl;  // string版本

// C++20：模板语法的lambda
auto generic20 = []<typename T>(T a, T b) {
    static_assert(is_arithmetic_v<T>);
    return a + b;
};
```

### 2.7 递归lambda（C++14）

```cpp
#include <iostream>
#include <functional>
using namespace std;

// 使用std::function实现递归lambda
function<int(int)> factorial = [&factorial](int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
};

// C++23: 使用auto参数的递归lambda（无类型擦除开销）
// auto fib = [](this auto&& self, int n) -> int {
//     return n <= 1 ? n : self(n-1) + self(n-2);
// };

int main() {
    cout << "5! = " << factorial(5) << endl;  // 120
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **值捕获的变量默认为const**：修改需加`mutable`
2. **引用捕获要注意被引用对象的生命周期**：lambda可能比被引用对象存活更久，导致悬垂引用
3. **this指针可通过`[this]`或`[*this]`（C++17）捕获**：`[this]`按引用捕获成员，`[*this]`按值拷贝整个对象
4. **空捕获列表`[]`的lambda可转换为函数指针**：有捕获则不行
5. **捕获的变量在lambda创建时确定值**：值捕获拷贝当时值，引用捕获绑定当时地址
6. **每个lambda表达式生成唯一的类型**：即使签名相同，两个lambda的类型也不同
7. **避免在循环中捕获引用的局部变量**：引用可能在循环迭代间失效
