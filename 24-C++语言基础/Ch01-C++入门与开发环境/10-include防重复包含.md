# include防重复包含

## 一、概念说明

当一个头文件被多次包含时，会导致**重定义**编译错误。头文件保护机制（Header Guard）确保头文件的内容在同一个编译单元中只被包含一次。

## 二、具体用法

### 2.1 传统方式：#ifndef

```cpp
// ===== student.h =====
#ifndef STUDENT_H       // 如果未定义STUDENT_H
#define STUDENT_H       // 定义STUDENT_H

#include <string>

class Student {
private:
    std::string name;
    int age;
public:
    Student(const std::string& n, int a);
    void display() const;
};

#endif // STUDENT_H
```

工作原理：
1. 预处理器首次遇到`#ifndef STUDENT_H`，条件为真，继续处理
2. 立即定义`#define STUDENT_H`
3. 如果同一编译单元再次包含此文件，`#ifndef STUDENT_H`条件为假，跳过整个文件内容

### 2.2 现代方式：#pragma once

```cpp
// ===== student.h =====
#pragma once

#include <string>

class Student {
private:
    std::string name;
    int age;
public:
    Student(const std::string& n, int a);
    void display() const;
};
```

`#pragma once`让编译器自行保证文件只被包含一次，代码更简洁。

### 2.3 两种方式对比

```cpp
// #ifndef 方式
#ifndef COMPLEX_PROJECT_UTILS_STRING_HELPER_H
#define COMPLEX_PROJECT_UTILS_STRING_HELPER_H

// 当路径复杂时，宏名称可能冲突
// 例如两个库都有 UTILS_H

#endif

// #pragma once 方式
#pragma once
// 编译器使用文件路径/inode判断，不会冲突
// 但不是C++标准的一部分
```

### 2.4 头文件包含链示例

```cpp
// ===== shape.h =====
#ifndef SHAPE_H
#define SHAPE_H

class Shape {
public:
    virtual double area() const = 0;
    virtual ~Shape() = default;
};

#endif

// ===== circle.h =====
#ifndef CIRCLE_H
#define CIRCLE_H

#include "shape.h"  // 包含基类

class Circle : public Shape {
    double radius;
public:
    Circle(double r);
    double area() const override;
};

#endif

// ===== main.cpp =====
#include "circle.h"   // 间接包含了shape.h
#include <iostream>

int main() {
    Circle c(5.0);
    std::cout << "面积: " << c.area() << std::endl;
    return 0;
}
```

输出：
```
面积: 78.5398
```

## 三、注意事项与常见陷阱

1. **宏名称必须唯一**：`#ifndef`的宏名应在整个项目中唯一，推荐使用`项目名_文件名_H`格式
2. **#pragma once非标准**：虽然被绝大多数编译器支持，但严格来说不是C++标准
3. **不要依赖隐式包含**：头文件应该包含它自己需要的所有头文件，不要假设调用者已经包含了
4. **两者可以共存**：为最大兼容性，可以同时使用两种方式
5. **源文件不需要保护**：`.cpp`文件不需要头文件保护，因为它们不会被`#include`
