# 类型识别 typeid

## 一、概念说明

`typeid`运算符返回表达式或类型的`type_info`对象，用于**运行时类型识别**（RTTI）。需要`#include <typeinfo>`。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
#include <typeinfo>
using namespace std;

class Base { public: virtual ~Base() = default; };
class Derived : public Base {};

int main() {
    // typeid用于类型
    cout << "int: " << typeid(int).name() << endl;
    cout << "double: " << typeid(double).name() << endl;
    cout << "string: " << typeid(string).name() << endl;

    // typeid用于表达式
    int x = 42;
    double y = 3.14;
    cout << "x的类型: " << typeid(x).name() << endl;
    cout << "x+y的类型: " << typeid(x + y).name() << endl;

    // typeid比较
    if (typeid(x) == typeid(int)) {
        cout << "x是int类型" << endl;
    }

    // 多态类型：运行时识别
    Base* bp = new Derived();
    cout << "静态类型: " << typeid(bp).name() << endl;      // Base*
    cout << "动态类型: " << typeid(*bp).name() << endl;     // Derived（需要虚函数）
    delete bp;

    return 0;
}
```

输出（GCC，名称是mangled）：
```
int: i
double: d
string: NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
x的类型: i
x+y的类型: d
x是int类型
静态类型: P4Base
动态类型: 7Derived
```

### 2.2 解析mangled名称

```cpp
#include <iostream>
#include <typeinfo>
#include <cxxabi.h>  // GCC/Clang

int main() {
    // 使用abi::__cxa_demangle解析名称
    int status;
    const char* name = typeid(std::string).name();
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0) {
        std::cout << "解析后: " << demangled << std::endl;
        free(demangled);
    }

    return 0;
}
```

输出：
```
解析后: std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >
```

## 三、注意事项与常见陷阱

1. **需要虚函数**：对多态类型，`typeid`需要至少一个虚函数才能运行时识别
2. **名称是mangled的**：GCC/Clang输出的名称是混淆的，需要demangle
3. **type_info不能拷贝**：只能比较`type_info`的地址或用`==`/`!=`
4. **性能开销**：RTTI有运行时开销，某些项目可能关闭（`-fno-rtti`）
5. **不要过度依赖typeid**：良好的多态设计不需要频繁检查类型
