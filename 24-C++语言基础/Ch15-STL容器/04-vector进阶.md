# vector进阶

## 一、概念说明

vector的进阶用法包括emplace系列、shrink_to_fit、swap技巧、以及异常安全保证等高级特性。这些特性在现代C++中对性能优化至关重要。

## 二、具体用法

### 2.1 emplace系列

```cpp
#include <vector>
#include <iostream>
#include <string>

struct Person {
    std::string name;
    int age;
    Person(std::string n, int a) : name(std::move(n)), age(a) {
        std::cout << "构造: " << name << std::endl;
    }
    Person(const Person& p) : name(p.name), age(p.age) {
        std::cout << "拷贝: " << name << std::endl;
    }
};

int main() {
    std::vector<Person> people;

    // push_back：先构造临时对象，再移动/拷贝
    people.push_back(Person("Alice", 25));  // 构造 + 移动

    // emplace_back：直接在容器中构造，无额外拷贝/移动
    people.emplace_back("Bob", 30);         // 只构造

    // emplace在指定位置构造
    people.emplace(people.begin(), "Charlie", 20);
}
```

### 2.2 shrink_to_fit

```cpp
void demonstrate_shrink() {
    std::vector<int> v;
    v.reserve(1000);
    for (int i = 0; i < 10; ++i) v.push_back(i);

    std::cout << "before: size=" << v.size()
              << " capacity=" << v.capacity() << std::endl;  // 10, 1000

    v.shrink_to_fit();  // 释放多余容量
    std::cout << "after: size=" << v.size()
              << " capacity=" << v.capacity() << std::endl;  // 10, 10

    // 注意：shrink_to_fit是非绑定请求，可能被忽略
}
```

### 2.3 swap技巧释放内存

```cpp
void release_memory() {
    std::vector<int> huge(1000000, 42);
    huge.clear();  // size=0，但capacity仍很大

    // swap技巧：彻底释放内存
    std::vector<int>().swap(huge);
    std::cout << "capacity after swap: " << huge.capacity() << std::endl;  // 0

    // C++11后可用shrink_to_fit替代
    // std::vector<int>(huge).swap(huge);
}
```

### 2.4 异常安全

```cpp
// push_back提供强异常安全保证
void exception_safety() {
    std::vector<std::string> v{"existing"};

    try {
        v.push_back("new");  // 如果扩容时抛异常，v保持原样
    } catch (const std::bad_alloc&) {
        std::cout << "分配失败，vector未修改" << std::endl;
    }

    // emplace_back同理
    try {
        v.emplace_back("another");
    } catch (...) {
        std::cout << "emplace失败" << std::endl;
    }
}
```

## 三、注意事项与常见陷阱

- `emplace_back`比`push_back`少一次移动/拷贝（对非平凡类型）
- `shrink_to_fit`是非绑定请求，编译器可能忽略
- swap技巧在C++11后不如以前常用（有`shrink_to_fit`）
- vector的异常安全：push/emplace提供强保证
- `reserve`不改变`size`，只改变`capacity`
- vector<bool>是特化的位压缩实现，行为不同于普通vector
