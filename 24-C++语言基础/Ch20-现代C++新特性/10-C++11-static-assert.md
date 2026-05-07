# C++11 static_assert

## 一、概念说明

`static_assert`是编译期断言，在编译时检查条件是否满足。不满足时产生编译错误并显示指定的消息。适用于模板约束、平台检查、类型尺寸验证等。

```cpp
#include <iostream>
#include <type_traits>

// 编译期检查类型大小
static_assert(sizeof(int) == 4, "int必须是4字节");
static_assert(sizeof(long long) >= 8, "long long至少8字节");

// 检查平台
static_assert(sizeof(void*) == 8, "仅支持64位平台");

int main() {
    // 函数内的static_assert
    constexpr int x = 42;
    static_assert(x == 42, "x应该等于42");

    std::cout << "所有静态断言通过" << std::endl;
    return 0;
}
```

**输出：**
```
所有静态断言通过
```

## 二、具体用法

### 2.1 模板中的static_assert

```cpp
#include <iostream>
#include <type_traits>

// 限制模板只接受算术类型
template <typename T>
T safeDivide(T a, T b) {
    static_assert(std::is_arithmetic<T>::value, "T必须是算术类型");
    static_assert(!std::is_same<T, bool>::value, "不支持bool类型");
    return a / b;
}

// 检查类型特征
template <typename T>
void process(T value) {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "T必须是数值类型");
    std::cout << "值: " << value << std::endl;
}

int main() {
    std::cout << "10/3=" << safeDivide(10, 3) << std::endl;
    std::cout << "10.0/3.0=" << safeDivide(10.0, 3.0) << std::endl;

    process(42);
    process(3.14);

    // safeDivide("a", "b"); // 编译错误：不满足static_assert
    // process("hello");      // 编译错误

    return 0;
}
```

**输出：**
```
10/3=3
10.0/3.0=3.33333
值: 42
值: 3.14
```

### 2.2 C++17的改进

```cpp
#include <iostream>
#include <type_traits>

// C++17: static_assert不需要错误消息
// static_assert(sizeof(int) == 4); // C++17起OK

// 但建议始终提供消息
static_assert(sizeof(int) == 4, "");

int main() {
    // 编译期布尔表达式
    constexpr bool is64bit = sizeof(void*) == 8;
    static_assert(is64bit, "需要64位平台");

    std::cout << "static_assert: 编译期断言" << std::endl;
    return 0;
}
```

**输出：**
```
static_assert: 编译期断言
```

## 三、注意事项与常见陷阱

- **`static_assert`必须使用常量表达式**：不能用运行时值。
- **断言失败产生编译错误**：不是运行时错误。
- **C++17起错误消息可省略**：但建议提供。
- **`static_assert`可以出现在类作用域中**：检查类的不变量。
- **与`assert`不同**：`assert`是运行时检查，`static_assert`是编译时检查。
