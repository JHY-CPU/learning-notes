# decltype类型推导

## 一、概念说明

`decltype`在**编译时**推导表达式的类型，但**不实际计算表达式的值**。与`auto`不同，`decltype`保留引用和const属性，适合在模板元编程和函数返回类型声明中使用。

## 二、具体用法

### 2.1 decltype基本用法

```cpp
#include <iostream>
#include <vector>
#include <type_traits>
using namespace std;

int main() {
    int x = 42;
    int& ref = x;
    const int cx = 100;

    // decltype保留精确类型
    decltype(x) a = 10;         // a是int
    decltype(ref) b = x;        // b是int&（保留引用）
    decltype(cx) c = 200;       // c是const int（保留const）

    // 与auto的对比
    auto d = ref;               // d是int（丢弃引用）
    decltype(ref) e = x;        // e是int&（保留引用）

    cout << "a: " << a << endl;
    b = 99;  // 通过引用修改x
    cout << "x: " << x << " (通过decltype引用修改)" << endl;
    // c = 300;  // 编译错误：c是const

    return 0;
}
```

输出：
```
a: 10
x: 99 (通过decltype引用修改)
```

### 2.2 decltype推导规则

```cpp
#include <iostream>
using namespace std;

int getValue() { return 42; }
int& getRef(int& x) { return x; }

int main() {
    // 规则1：decltype(变量) → 变量的声明类型
    int x = 10;
    decltype(x) a;             // int

    // 规则2：decltype(表达式) → 表达式的类型
    decltype(x + 1) b;         // int（x+1是prvalue，类型int）
    decltype((x)) c = x;       // int&（(x)是lvalue表达式，类型int&）

    // 规则3：decltype(函数调用) → 函数返回类型
    decltype(getValue()) d;    // int
    decltype(getRef(x)) e = x; // int&

    // 注意：加括号的差异
    decltype(x) f;             // int（变量）
    decltype((x)) g = x;       // int&（表达式，lvalue→引用）

    cout << "sizeof(a): " << sizeof(a) << endl;
    cout << "sizeof(c): " << sizeof(c) << endl;

    return 0;
}
```

输出：
```
sizeof(a): 4
sizeof(c): 8  // 引用的sizeof是指针大小
```

### 2.3 decltype(auto)（C++14）

```cpp
#include <iostream>
#include <string>
using namespace std;

string& getStringRef(string& s) { return s; }

// 使用decltype(auto)作为返回类型
// 自动推导并保留引用属性
decltype(auto) forwardRef(string& s) {
    return getStringRef(s);  // 返回string&而非string
}

// 对比普通auto返回类型
auto forwardWithAuto(string& s) -> decltype(getStringRef(s)) {
    return getStringRef(s);
}

int main() {
    string original = "Hello";

    // decltype(auto)保留了引用
    decltype(auto) result = forwardRef(original);
    result = "Modified";  // 修改了original

    cout << "original: " << original << endl;

    // 用于变量声明
    int x = 42;
    int& ref = x;

    decltype(auto) y = ref;    // y是int&（保留引用）
    auto z = ref;              // z是int（丢弃引用）

    y = 99;
    cout << "x: " << x << ", z: " << z << endl;

    return 0;
}
```

输出：
```
original: Modified
x: 99, z: 42
```

### 2.4 decltype在模板中的应用

```cpp
#include <iostream>
#include <vector>
using namespace std;

// C++11：使用decltype推导模板函数返回类型
template<typename Container, typename Index>
auto getElement(Container& c, Index i) -> decltype(c[i]) {
    return c[i];
}

// C++14：直接使用auto返回类型
template<typename Container, typename Index>
decltype(auto) getElement14(Container& c, Index i) {
    return c[i];
}

// 推导表达式类型用于声明变量
template<typename T, typename U>
auto add(const T& a, const U& b) -> decltype(a + b) {
    return a + b;
}

int main() {
    vector<int> nums = {10, 20, 30, 40, 50};

    // getElement返回引用，可以修改容器元素
    getElement(nums, 2) = 999;
    cout << "nums[2]: " << nums[2] << endl;

    // 自动推导加法结果类型
    auto result = add(3, 4.5);  // double
    cout << "3 + 4.5 = " << result << endl;

    return 0;
}
```

输出：
```
nums[2]: 999
3 + 4.5 = 7.5
```

## 三、注意事项与常见陷阱

1. **括号影响推导**：`decltype(x)`和`decltype((x))`结果可能不同，后者总是引用类型
2. **decltype不求值**：`decltype(expr)`只分析类型，不执行表达式
3. **auto vs decltype**：auto从初始化推导（丢弃引用），decltype从表达式推导（保留精确类型）
4. **decltype(auto)**：C++14特性，结合了auto的简洁和decltype的精确推导
5. **SFINAE应用**：decltype常用于模板的SFINAE（替换失败不是错误）技术中
