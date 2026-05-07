# Rule of Zero

## 一、概念说明

**Rule of Zero**（零法则）是现代C++的核心设计原则：**如果你的类不需要直接管理资源，就不应该定义任何特殊成员函数**（析构、拷贝构造、拷贝赋值、移动构造、移动赋值）。让编译器自动生成这些函数即可。

### 1.1 核心思想

使用RAII包装类（如`std::unique_ptr`、`std::vector`、`std::string`）管理资源，而不是裸指针和手动`new/delete`。这样编译器自动生成的特殊成员函数就是正确且高效的。

## 二、具体用法

### 2.1 遵循Rule of Zero的类

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>

class Employee {
private:
    std::string name;                    // string管理自己的内存
    std::vector<std::string> skills;     // vector管理自己的内存
    std::unique_ptr<int> employeeId;     // unique_ptr管理动态内存
public:
    Employee(const std::string& n, int id)
        : name(n), employeeId(std::make_unique<int>(id)) {}

    // 无需定义任何特殊成员函数！
    // 编译器自动生成正确的析构、移动操作
    // 拷贝操作被unique_ptr阻止（正确行为）

    void addSkill(const std::string& skill) {
        skills.push_back(skill);
    }

    void display() const {
        std::cout << name << " (ID: " << *employeeId << "), 技能: ";
        for (const auto& s : skills) std::cout << s << " ";
        std::cout << std::endl;
    }
};

int main() {
    Employee e1("张三", 1001);
    e1.addSkill("C++");
    e1.addSkill("Python");
    e1.display();

    Employee e2 = std::move(e1);  // 移动OK
    e2.display();

    // Employee e3 = e2;  // 编译错误！unique_ptr不可拷贝
    return 0;
}
```

**输出：**
```
张三 (ID: 1001), 技能: C++ Python
张三 (ID: 1001), 技能: C++ Python
```

### 2.2 对比：违反Rule of Zero的类

```cpp
#include <iostream>
#include <cstring>

// 不推荐：手动管理资源，需要定义所有特殊成员
class BadString {
private:
    char* data;
public:
    BadString(const char* s) : data(new char[strlen(s) + 1]) {
        strcpy(data, s);
    }
    ~BadString() { delete[] data; }
    BadString(const BadString& o) : data(new char[strlen(o.data) + 1]) {
        strcpy(data, o.data);
    }
    BadString& operator=(const BadString& o) {
        if (this != &o) {
            delete[] data;
            data = new char[strlen(o.data) + 1];
            strcpy(data, o.data);
        }
        return *this;
    }
};

// 推荐：使用std::string，遵循Rule of Zero
class GoodString {
    std::string data;  // string管理一切
public:
    GoodString(const char* s) : data(s) {}
    // 无需定义任何特殊成员函数
};
```

## 三、注意事项与常见陷阱

- 优先使用标准库容器和智能指针管理资源
- 如果类包含`std::unique_ptr`成员，拷贝操作自动被禁止（符合预期）
- 如果需要可拷贝的资源管理，使用`std::shared_ptr`而非裸指针
- Rule of Zero是最高优先级的设计目标
- 只有当RAII包装类无法满足需求时，才回退到Rule of Three/Five
