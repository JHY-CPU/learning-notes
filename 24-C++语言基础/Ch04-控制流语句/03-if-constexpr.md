# if-constexpr

## 一、概念说明

`if constexpr`（C++17）是**编译期条件分支**。编译器只编译条件为真的分支代码，丢弃假的分支。这在模板编程中特别有用，可以避免实例化不支持的代码。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
#include <type_traits>
#include <string>
using namespace std;

template<typename T>
void printValue(const T& val) {
    if constexpr (is_integral_v<T>) {
        cout << "整数: " << val << endl;
    } else if constexpr (is_floating_point_v<T>) {
        cout << "浮点: " << val << endl;
    } else if constexpr (is_same_v<T, string>) {
        cout << "字符串: " << val << endl;
    } else {
        cout << "其他类型" << endl;
    }
}

int main() {
    printValue(42);         // 整数
    printValue(3.14);       // 浮点
    printValue("hello"s);   // 字符串
    printValue(true);       // 整数（bool是整数类型）

    return 0;
}
```

输出：
```
整数: 42
浮点: 3.14
字符串: hello
整数: 1
```

### 2.2 在模板中避免实例化错误

```cpp
#include <iostream>
#include <vector>
#include <type_traits>
using namespace std;

template<typename T>
auto getFirst(const T& container) {
    if constexpr (is_array_v<remove_reference_t<T>>) {
        return container[0];  // C风格数组
    } else {
        return container.front();  // STL容器
    }
}

int main() {
    vector<int> vec = {1, 2, 3};
    int arr[] = {10, 20, 30};

    cout << "vector第一个: " << getFirst(vec) << endl;
    cout << "数组第一个: " << getFirst(arr) << endl;

    return 0;
}
```

输出：
```
vector第一个: 1
数组第一个: 10
```

## 三、注意事项与常见陷阱

1. **条件必须是编译期常量**：`if constexpr (x > 0)`中x必须是constexpr
2. **被丢弃的分支仍需语法正确**：不被选择的分支不能有语法错误
3. **短路行为**：编译器不检查被丢弃分支中的语义错误
4. **替代SFINAE**：比传统的`enable_if`更直观
5. **不能在非模板函数中依赖**：普通函数中的`if constexpr`用处有限
