# C++11 auto与decltype

## 一、概念说明

`auto`让编译器自动推导变量类型，`decltype`获取表达式的类型。它们是现代C++类型推导的基石，大幅减少冗长的类型声明。

- `auto`：从初始化表达式推导类型（C++11 §7.1.6.4）
- `decltype(expr)`：获取表达式的精确类型（含引用、const限定符）（C++11 §7.1.6.2）

### 1.1 核心区别

| 特性 | auto | decltype |
|------|------|---------|
| 推导来源 | 初始化表达式 | 表达式本身 |
| 引用处理 | 去除引用 | 保留引用 |
| const处理 | 去除顶层const | 保留const |
| 用例 | 变量声明 | 尾置返回类型、类型萃取 |

```cpp
#include <iostream>
#include <vector>
#include <type_traits>

int main() {
    // auto推导
    auto i = 42;           // int
    auto d = 3.14;         // double
    auto s = "hello";      // const char*
    auto v = std::vector<int>{1, 2, 3}; // std::vector<int>

    // decltype获取类型
    decltype(i) j = 100;   // int
    decltype((i)) k = i;   // int&（因为(i)是左值表达式）

    // 类型检查
    std::cout << "i is int: " << std::is_same<decltype(i), int>::value << std::endl;
    std::cout << "d is double: " << std::is_same<decltype(d), double>::value << std::endl;
    std::cout << "k is int&: " << std::is_same<decltype(k), int&>::value << std::endl;

    return 0;
}
```

**输出：**
```
i is int: 1
d is double: 1
k is int&: 1
```

## 二、具体用法

### 2.1 auto的推导规则

auto的推导规则与模板参数推导类似：去掉引用和顶层const。

```cpp
#include <iostream>
#include <type_traits>

int main() {
    int x = 42;
    const int cx = 42;
    const int& cr = x;

    auto a1 = x;    // int（去掉引用和const）
    auto a2 = cx;   // int（去掉const）
    auto a3 = cr;   // int（去掉引用和const）
    auto& a4 = cx;  // const int&（保留const，引用需显式声明）

    // 指针的auto
    const int* cp = &cx;
    auto p1 = cp;       // const int*（指针本身不是顶层const）
    auto* p2 = cp;      // const int*

    // auto不能推导的场景
    // auto arr[] = {1, 2, 3};       // 错误：不能推导数组
    // auto func(int x);             // 错误：不能用于函数参数（C++14前）
    // std::vector<auto> v;          // 错误：不能用于模板参数

    std::cout << "a1 is int: " << std::is_same<decltype(a1), int>::value << std::endl;
    std::cout << "a4 is const int&: " << std::is_same<decltype(a4), const int&>::value << std::endl;

    return 0;
}
```

### 2.2 decltype的推导规则

decltype精确保留表达式的类型，包括引用和const。

```cpp
#include <iostream>
#include <type_traits>
#include <vector>

// decltype用于返回类型推导（C++11尾置返回类型）
template <typename Container>
auto getItem(Container& c, size_t i) -> decltype(c[i]) {
    return c[i];
}

// C++14简化版
template <typename Container>
decltype(auto) getItem2(Container& c, size_t i) {
    return c[i];
}

int main() {
    std::vector<int> v = {1, 2, 3};
    getItem(v, 0) = 42; // 返回int&，可以赋值
    std::cout << "v[0] = " << v[0] << std::endl;

    // decltype vs auto
    const int ci = 0;
    auto a = ci;          // int（去const）
    decltype(ci) di = 0;  // const int（保留const）

    // decltype((x))得到引用
    int x = 10;
    decltype(x) dx = x;     // int
    decltype((x)) rx = x;   // int&（括号使x成为左值表达式）

    std::cout << "dx is int: " << std::is_same<decltype(dx), int>::value << std::endl;
    std::cout << "rx is int&: " << std::is_same<decltype(rx), int&>::value << std::endl;

    return 0;
}
```

**输出：**
```
v[0] = 42
dx is int: 1
rx is int&: 1
```

### 2.3 decltype(auto)（C++14）

```cpp
#include <iostream>
#include <string>

// 完美转发返回值
template <typename F, typename Arg>
decltype(auto) call_and_log(F&& f, Arg&& arg) {
    std::cout << "调用函数..." << std::endl;
    return std::forward<F>(f)(std::forward<Arg>(arg));
}

std::string get_name() { return "Alice"; }
std::string& get_ref(std::string& s) { return s; }

int main() {
    // decltype(auto)保留返回类型的引用性
    std::string name = "Bob";
    decltype(auto) ref = get_ref(name); // std::string&
    ref = "Charlie";
    std::cout << "name = " << name << std::endl; // Charlie

    return 0;
}
```

**输出：**
```
name = Charlie
```

## 三、注意事项与常见陷阱

1. **`auto`去掉引用和const**：需要保留时用`auto&`或`const auto&`。
2. **`decltype((x))`得到引用**：括号使其成为左值表达式，推导为引用类型。
3. **`auto`不能用于函数参数**（C++14前）：C++14起可用于泛型lambda参数。
4. **`auto x{1}`在C++11中推导为`initializer_list<int>`**：C++17改为`int`，统一初始化行为。
5. **`decltype`在编译期求值**：不执行表达式，只分析类型。
6. **`decltype(auto)`是C++14特性**：简化同时需要`auto`便利和`decltype`精确性的场景。
