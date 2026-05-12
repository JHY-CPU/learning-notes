# vector进阶

## 一、概念说明

vector的进阶用法包括emplace系列、shrink_to_fit、swap技巧、异常安全保证以及`vector<bool>`特化等高级特性。这些特性在现代C++中对性能优化至关重要。理解vector的内部机制（如扩容策略、迭代器失效）是编写高效代码的基础。

### 1.1 扩容策略

| 实现 | 扩容因子 | 说明 |
|------|---------|------|
| GCC libstdc++ | 2x | 每次容量翻倍 |
| MSVC | 1.5x | 每次增加50% |
| Clang libc++ | 2x | 每次容量翻倍 |

扩容因子影响均摊复杂度和内存利用率：2x更快但更费内存，1.5x更省内存但扩容更频繁。

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
    Person(Person&& p) noexcept : name(std::move(p.name)), age(p.age) {
        std::cout << "移动: " << name << std::endl;
    }
};

int main() {
    std::vector<Person> people;
    people.reserve(10);  // 避免扩容干扰观察

    // push_back：先构造临时对象，再移动
    people.push_back(Person("Alice", 25));  // 构造临时 + 移动 = 2次操作

    // emplace_back：直接在容器中构造
    people.emplace_back("Bob", 30);         // 只构造 = 1次操作

    // emplace在指定位置构造
    people.emplace(people.begin(), "Charlie", 20);

    // emplace返回新元素的引用（C++17）
    auto& ref = people.emplace_back("Dave", 35);
    ref.age = 36;  // 直接修改
}
```

### 2.2 shrink_to_fit与内存管理

```cpp
void memory_management() {
    std::vector<int> v;
    v.reserve(10000);
    for (int i = 0; i < 100; ++i) v.push_back(i);

    std::cout << "size=" << v.size()          // 100
              << " capacity=" << v.capacity()  // 10000
              << std::endl;

    // shrink_to_fit：请求释放多余容量（非绑定）
    v.shrink_to_fit();
    std::cout << "capacity=" << v.capacity()   // 可能是100
              << std::endl;

    // C++11之前的swap技巧
    std::vector<int>(v).swap(v);  // 创建临时vector并交换

    // 彻底释放内存
    std::vector<int>().swap(v);   // capacity归零
}
```

### 2.3 异常安全

```cpp
#include <stdexcept>

class Resource {
public:
    Resource(int v) {
        if (v < 0) throw std::invalid_argument("负值");
    }
};

void exception_safety() {
    std::vector<Resource> v;
    v.emplace_back(1);
    v.emplace_back(2);

    try {
        v.emplace_back(-3);  // 构造时抛异常
    } catch (...) {
        // 强异常安全：v保持{1, 2}不变
        std::cout << "大小: " << v.size() << std::endl;  // 2
    }

    // 移动构造noexcept的重要性：
    // 扩容时，若移动构造是noexcept则用移动，否则用拷贝
    // 因为移动中途抛异常无法回滚
}
```

### 2.4 vector<bool>特化

```cpp
#include <vector>
#include <iostream>

void vector_bool_demo() {
    std::vector<bool> flags = {true, false, true, false, true};

    // 特化：位压缩存储，每个bool占1位
    std::cout << "sizeof(vector<bool>): " << sizeof(flags) << std::endl;

    // 注意：flags[i]返回代理对象，不是bool&
    // auto& ref = flags[0];  // 编译错误！
    auto ref = flags[0];     // OK，值拷贝

    // flip翻转
    flags.flip();  // 所有位取反

    // 问题：不能取地址，不能绑定引用
    // bool* p = &flags[0];  // 编译错误

    // 替代方案：用vector<char>代替
    std::vector<char> char_flags = {1, 0, 1};
    char& cref = char_flags[0];  // OK
}
```

### 2.5 高级操作

```cpp
#include <algorithm>

void advanced_ops() {
    std::vector<int> v = {5, 3, 1, 4, 2};

    // 排序
    std::sort(v.begin(), v.end());

    // 二分查找（需已排序）
    bool found = std::binary_search(v.begin(), v.end(), 3);

    // 批量赋值
    std::fill(v.begin(), v.end(), 0);

    // resize + 默认值
    v.resize(10, 42);  // 扩充到10，新元素为42

    // 初始化列表赋值（C++11）
    v = {1, 2, 3, 4, 5};

    // swap（O(1)，交换内部指针）
    std::vector<int> other = {10, 20, 30};
    v.swap(other);
}
```

## 三、注意事项与常见陷阱

1. **`emplace_back`比`push_back`少一次移动/拷贝**：对非平凡类型有显著性能差异
2. **`shrink_to_fit`是非绑定请求**：编译器可能忽略，不保证释放
3. **`vector<bool>`是特化**：位压缩，不满足标准容器的所有要求，不能取地址或绑定引用
4. **扩容策略影响性能**：频繁push_back时reserve可避免多次扩容
5. **移动构造的noexcept**：影响扩容时选择移动还是拷贝，务必标记noexcept
6. **`data()`返回连续内存**：C++11后保证，可安全传给C接口
7. **迭代器失效**：任何可能导致扩容的操作都会使所有迭代器失效
