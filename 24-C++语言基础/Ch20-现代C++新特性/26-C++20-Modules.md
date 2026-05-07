# C++20 Modules

## 一、概念说明

Modules替代传统的头文件包含机制，提供更快的编译速度、更好的封装性、消除头文件的宏泄漏问题。

核心概念：
- `export module`：定义模块
- `import`：导入模块
- `export`：导出声明

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

```cpp
// 模块分区（概念性）
// export module math:algebra;
// export int gcd(int a, int b) { ... }

// export module math:geometry;
// export double circleArea(double r) { return pi * r * r; }

// export module math;
// export import :algebra;
// export import :geometry;

// 使用
// import math;
// auto area = circleArea(5.0);
```

### 2.2 头文件单元

```cpp
// 头文件单元：渐进式迁移
// import <iostream>;     // 标准库头文件单元
// import <vector>;
// import "myheader.h";  // 传统头文件作为模块导入

import std; // C++23: 导入整个标准库

int main() {
    std::cout << "Hello Modules!" << std::endl;
    return 0;
}
```

## 三、注意事项与常见陷阱

- **编译器支持尚不完全**：MSVC支持最好，GCC/Clang逐步完善。
- **模块接口文件扩展名因编译器而异**：`.cppm`、`.ixx`等。
- **模块中的宏不导出**：这是设计目标，消除宏污染。
- **`import std;`是C++23特性**：导入整个标准库。
- **模块和头文件可以共存**：渐进式迁移。
- **模块的编译顺序很重要**：模块接口需先编译。
