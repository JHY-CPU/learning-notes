# sizeof运算符

## 一、概念说明

`sizeof`是编译期运算符，返回类型或表达式所占的**字节数**。它不计算表达式的值，只分析类型信息。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
using namespace std;

int main() {
    // sizeof用于类型（必须加括号）
    cout << "sizeof(int): " << sizeof(int) << endl;
    cout << "sizeof(double): " << sizeof(double) << endl;

    // sizeof用于变量（括号可选）
    int x = 42;
    cout << "sizeof x: " << sizeof x << endl;
    cout << "sizeof(x): " << sizeof(x) << endl;

    // sizeof数组 = 元素大小 × 元素个数
    int arr[10];
    cout << "sizeof(arr): " << sizeof(arr) << endl;              // 40
    cout << "元素个数: " << sizeof(arr) / sizeof(arr[0]) << endl; // 10

    // 注意：函数参数中的数组退化为指针
    auto func = [](int arr[]) {
        cout << "函数内sizeof(arr): " << sizeof(arr) << endl;  // 指针大小
    };
    func(arr);

    return 0;
}
```

输出（64位系统）：
```
sizeof(int): 4
sizeof(double): 8
sizeof x: 4
sizeof(x): 4
sizeof(arr): 40
元素个数: 10
函数内sizeof(arr): 8
```

### 2.2 sizeof不求值

```cpp
#include <iostream>
using namespace std;

int sideEffect() {
    cout << "sideEffect被调用！" << endl;
    return 42;
}

int main() {
    // sizeof不实际计算表达式的值
    // 所以副作用不会发生
    int x = sizeof(sideEffect());  // sideEffect不会被调用！
    cout << "x = " << x << endl;   // 4（int的大小）

    // sizeof是编译期运算符
    // 以下sizeof的结果在编译时就确定了
    int a[100];
    cout << "数组大小: " << sizeof(a) << endl;  // 400

    return 0;
}
```

输出：
```
x = 4
数组大小: 400
```

### 2.3 sizeof与结构体

```cpp
#include <iostream>
using namespace std;

struct A {
    char c;     // 1字节
    int i;      // 4字节
    char d;     // 1字节
};  // 实际大小12字节（填充对齐）

struct B {
    int i;      // 4字节
    char c;     // 1字节
    char d;     // 1字节
};  // 实际大小8字节

struct C {
    char c;
};  // 实际大小1字节

int main() {
    cout << "sizeof(A): " << sizeof(A) << endl;  // 12
    cout << "sizeof(B): " << sizeof(B) << endl;  // 8
    cout << "sizeof(C): " << sizeof(C) << endl;  // 1

    // C++11空基类优化
    struct Empty {};
    struct Derived : Empty {
        int x;
    };
    cout << "sizeof(Empty): " << sizeof(Empty) << endl;      // 1
    cout << "sizeof(Derived): " << sizeof(Derived) << endl;  // 4（空基类优化）

    return 0;
}
```

输出：
```
sizeof(A): 12
sizeof(B): 8
sizeof(C): 1
sizeof(Empty): 1
sizeof(Derived): 4
```

## 三、注意事项与常见陷阱

1. **sizeof不求值**：`sizeof(f())`不调用f，只返回f返回类型的大小
2. **数组退化**：函数参数中的数组退化为指针，sizeof得到指针大小
3. **结构体填充**：sizeof包含填充字节，大小可能比成员大小之和大
4. **sizeof(char)始终为1**：C++标准规定char的大小为1字节
5. **64位系统指针大小为8**：不要假设指针大小等于int大小
