# C++20 Modules

## 一、概念说明

Modules（C++20 §10）替代传统的头文件包含机制（`#include`），提供更快的编译速度、更好的封装性、消除头文件的宏泄漏问题。Modules是C++20四大特性之一。

### 1.1 头文件的问题

| 问题 | 说明 |
|------|------|
| 编译慢 | 每个翻译单元重复解析头文件 |
| 宏泄漏 | 头文件中的宏污染所有包含者 |
| 封装差 | 无法真正隐藏实现细节 |
| 依赖管理复杂 | `#pragma once`/include guard |

### 1.2 Modules的核心概念

```
export module math;     // 定义模块
import math;            // 导入模块
export int add(int,int); // 导出声明
```

```cpp
// math.cppm（模块接口文件）
export module math;

export int add(int a, int b) {
    return a + b;
}

export constexpr double pi = 3.14159265358979;

export template <typename T>
T square(T x) {
    return x * x;
}
```

```cpp
// main.cpp
import math;
import <iostream>;

int main() {
    std::cout << "add(3,4)=" << add(3, 4) << std::endl;
    std::cout << "pi=" << pi << std::endl;
    std::cout << "square(5)=" << square(5) << std::endl;
    return 0;
}
```

**输出：**
```
add(3,4)=7
pi=3.14159
square(5)=25
```

## 二、具体用法

### 2.1 模块分区

大型模块可以分成多个分区文件。

```cpp
// 模块分区（概念性示例）
// math:algebra.cppm
export module math:algebra;
export int gcd(int a, int b) { while(b) { int t=b; b=a%b; a=t; } return a; }

// math:geometry.cppm
export module math:geometry;
export double circleArea(double r) { return 3.14159 * r * r; }

// math.cppm（主模块）
export module math;
export import :algebra;
export import :geometry;
```

```cpp
// 使用
import math;
auto area = circleArea(5.0);
auto g = gcd(48, 18);
```

### 2.2 头文件单元

渐进式迁移：传统头文件可以作为模块导入。

```cpp
// 头文件单元：渐进式迁移
import <iostream>;     // 标准库头文件单元
import <vector>;
import <string>;

// import "myheader.h";  // 传统头文件作为模块导入

int main() {
    std::vector<int> v = {1, 2, 3};
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;
    return 0;
}
```

### 2.3 模块的封装

```cpp
// module.cppm
export module logger;

// 不导出：内部实现
namespace detail {
    void writeLog(const char* msg) {
        // 内部实现，外部不可见
    }
}

// 导出：公开接口
export void log(const char* msg) {
    detail::writeLog(msg);
}

// 导出类
export class Logger {
public:
    void info(const char* msg);
    void error(const char* msg);
private:
    // 实现细节对外隐藏
    int level_ = 0;
};
```

## 三、注意事项与常见陷阱

1. **编译器支持尚不完全**：MSVC支持最好，GCC/Clang逐步完善。
2. **模块接口文件扩展名因编译器而异**：`.cppm`（GCC/Clang）、`.ixx`（MSVC）。
3. **模块中的宏不导出**：这是设计目标，消除宏污染。
4. **`import std;`是C++23特性**：导入整个标准库，需要编译器支持。
5. **模块和头文件可以共存**：支持渐进式迁移。
6. **模块的编译顺序很重要**：模块接口需先编译，构建系统需要感知模块依赖。
7. **`export`不能用于局部声明**：只能用于命名空间作用域。
