# C++11 static_assert

## 一、概念说明

`static_assert`（C++11 §7.4）是编译期断言，在编译时检查条件是否满足。不满足时产生编译错误并显示指定的消息。与运行时`assert`不同，`static_assert`在编译阶段就阻止不正确的代码。

### 1.1 static_assert vs assert

| 特性 | static_assert | assert |
|------|--------------|--------|
| 检查时机 | 编译时 | 运行时 |
| 条件类型 | 常量表达式 | 任意布尔表达式 |
| 头文件 | 无（语言特性） | `<cassert>` |
| Release | 始终检查 | 可能被禁用 |

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
#include <vector>

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

// 检查容器元素类型
template <typename Container>
void printContainer(const Container& c) {
    using Elem = typename Container::value_type;
    static_assert(std::is_arithmetic<Elem>::value, "容器元素必须是算术类型");
    for (const auto& x : c) std::cout << x << " ";
    std::cout << std::endl;
}

int main() {
    std::cout << "10/3=" << safeDivide(10, 3) << std::endl;
    std::cout << "10.0/3.0=" << safeDivide(10.0, 3.0) << std::endl;

    process(42);
    process(3.14);

    std::vector<int> v = {1, 2, 3};
    printContainer(v);

    // safeDivide("a", "b"); // 编译错误：不满足static_assert
    // process("hello");      // 编译错误
    // std::vector<std::string> vs = {"a"};
    // printContainer(vs);    // 编译错误

    return 0;
}
```

**输出：**
```
10/3=3
10.0/3.0=3.33333
值: 42
值: 3.14
1 2 3
```

### 2.2 编译期计算验证

```cpp
#include <iostream>

// 编译期验证数据结构大小
template <typename T>
struct AlignedBuffer {
    T data[100];
};
static_assert(sizeof(AlignedBuffer<int>) == 400, "数组大小不正确");

// 验证类布局
struct PackedStruct {
    char a;
    int b;
    char c;
};
// static_assert(sizeof(PackedStruct) == 5, "可能存在padding"); // 可能失败

// 编译期数学验证
constexpr int factorial(int n) {
    int r = 1;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}
static_assert(factorial(5) == 120, "5!应该等于120");
static_assert(factorial(0) == 1, "0!应该等于1");

int main() {
    std::cout << "static_assert: 编译期断言工具" << std::endl;
    std::cout << "sizeof(AlignedBuffer<int>) = " << sizeof(AlignedBuffer<int>) << std::endl;
    return 0;
}
```

**输出：**
```
static_assert: 编译期断言工具
sizeof(AlignedBuffer<int>) = 400
```

### 2.3 C++17改进

C++17起`static_assert`的错误消息可以省略（但仍建议提供）。

```cpp
#include <iostream>
#include <type_traits>

// C++17: static_assert不需要错误消息
// static_assert(sizeof(int) == 4); // C++17起OK

// 但建议始终提供消息
static_assert(sizeof(int) == 4, "int应为4字节");

// 用constexpr变量简化
constexpr bool is64bit = sizeof(void*) == 8;
static_assert(is64bit, "需要64位平台");

int main() {
    std::cout << "平台验证通过" << std::endl;
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`static_assert`必须使用常量表达式**：不能用运行时值作为条件。
2. **断言失败产生编译错误**：不是运行时错误，编译即停止。
3. **C++17起错误消息可省略**：但建议提供以提高可读性。
4. **`static_assert`可以出现在类作用域中**：检查类的不变量和模板参数。
5. **与`assert`不同**：`assert`是运行时检查，`static_assert`是编译时检查。
6. **`static_assert`对模板非常有用**：在实例化时提供清晰的错误信息。
