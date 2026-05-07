# new与delete详解

## 一、概念说明

`new`运算符在堆上分配内存并调用构造函数，`delete`调用析构函数并释放内存。与C语言的`malloc`/`free`不同，`new`/`delete`会正确调用对象的构造和析构函数。

单对象形式：`new T` / `delete ptr`
数组形式：`new T[n]` / `delete[] ptr`

```cpp
#include <iostream>
#include <string>

class Student {
    std::string name;
    int age;
public:
    Student(std::string n, int a) : name(n), age(a) {
        std::cout << "构造: " << name << std::endl;
    }
    ~Student() {
        std::cout << "析构: " << name << std::endl;
    }
    void show() const {
        std::cout << name << ", " << age << "岁" << std::endl;
    }
};

int main() {
    // 单对象
    Student* s1 = new Student("张三", 20);
    s1->show();
    delete s1;

    // 数组
    Student* arr = new Student[2]{
        Student("李四", 21),
        Student("王五", 22)
    };
    arr[0].show();
    arr[1].show();
    delete[] arr; // 必须用delete[]

    return 0;
}
```

**输出：**
```
构造: 张三
张三, 20岁
析构: 张三
构造: 李四
构造: 王五
李四, 21岁
王五, 22岁
析构: 王五
析构: 李四
```

## 二、具体用法

### 2.1 new的多种初始化方式

```cpp
#include <iostream>

int main() {
    // 默认初始化（值不确定）
    int* p1 = new int;

    // 值初始化（int为0）
    int* p2 = new int();

    // 直接初始化
    int* p3 = new int(42);

    // 列表初始化（C++11）
    int* p4 = new int{42};

    std::cout << "*p1 (未初始化): " << *p1 << std::endl;
    std::cout << "*p2 (值初始化): " << *p2 << std::endl;
    std::cout << "*p3 (直接初始化): " << *p3 << std::endl;
    std::cout << "*p4 (列表初始化): " << *p4 << std::endl;

    delete p1; delete p2; delete p3; delete p4;
    return 0;
}
```

**输出：**
```
*p1 (未初始化): 0
*p2 (值初始化): 0
*p3 (直接初始化): 42
*p4 (列表初始化): 42
```

### 2.2 new与malloc的区别

| 特性 | new | malloc |
|------|-----|--------|
| 调用构造函数 | 是 | 否 |
| 返回类型 | 类型指针 | void* |
| 失败行为 | 抛异常 | 返回NULL |
| 可重载 | 是 | 否 |
| 计算大小 | 自动 | 手动指定 |

## 三、注意事项与常见陷阱

- **`new[]`必须配`delete[]`**：用`delete`释放数组会导致未定义行为。
- **`new`单对象配`delete`**：不能用`delete[]`释放单对象。
- **裸指针容易泄漏**：异常时可能跳过`delete`，应使用智能指针。
- **`delete`后置空**：`delete ptr; ptr = nullptr;`防止重复释放。
- **构造函数抛异常**：`new`会在构造函数抛异常时自动释放已分配的内存。
