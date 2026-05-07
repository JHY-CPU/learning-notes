# 异常与STL

## 一、概念说明

STL容器和算法在异常安全方面有明确的保证。了解STL的异常行为有助于编写异常安全的代码。不同操作提供不同级别的异常安全保证。

## 二、具体用法

### 2.1 vector扩容异常安全

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class Widget {
    int value;
public:
    Widget(int v) : value(v) {
        if (v < 0) throw std::invalid_argument("负值");
    }
    int get() const { return value; }
};

int main() {
    std::vector<Widget> vec;
    try {
        vec.push_back(Widget(1));
        vec.push_back(Widget(2));
        vec.push_back(Widget(-3));  // 抛异常
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        // vec中已有的元素仍然有效
        std::cout << "已保存元素: " << vec.size() << std::endl;
    }
}
```

### 2.2 容器操作的异常保证

```cpp
/*
| 操作               | 异常安全保证    |
|-------------------|---------------|
| vector::push_back  | 强保证         |
| vector::insert     | 强保证         |
| vector::emplace    | 强保证         |
| vector::erase      | 不抛异常       |
| list::push_back    | 强保证         |
| map::insert        | 强保证         |
| map::operator[]    | 基本保证       |
| sort               | 基本保证       |
*/
```

### 2.3 异常安全的元素插入

```cpp
void safe_insert() {
    std::vector<std::string> vec{"hello", "world"};

    try {
        // emplace提供强异常安全
        vec.emplace(vec.begin(), "new");
        std::cout << "插入成功: " << vec[0] << std::endl;
    } catch (const std::bad_alloc& e) {
        // vector保持原样
        std::cerr << "内存不足，vector未修改" << std::endl;
    }
}
```

### 2.4 算法异常安全

```cpp
#include <algorithm>

class MayThrow {
    int value;
public:
    MayThrow(int v) : value(v) {}
    bool operator<(const MayThrow& other) const {
        if (value < 0) throw std::runtime_error("比较异常");
        return value < other.value;
    }
    int get() const { return value; }
};

void safe_algorithm() {
    std::vector<MayThrow> vec{3, 1, 4, 1, 5};
    try {
        std::sort(vec.begin(), vec.end());
    } catch (...) {
        // sort提供基本保证：vector可能处于未排序状态
        // 但不会泄漏资源
        std::cerr << "排序异常" << std::endl;
    }
}
```

### 2.5 allocator异常

```cpp
// 自定义allocator可能抛出bad_alloc
void allocator_demo() {
    try {
        std::vector<int> huge(10000000000LL);  // 可能失败
    } catch (const std::bad_alloc& e) {
        std::cerr << "分配失败: " << e.what() << std::endl;
    }
}
```

## 三、注意事项与常见陷阱

- vector的`push_back`、`emplace_back`提供强异常安全
- 排序算法（`sort`）只提供基本保证
- 容器析构函数不会抛异常（noexcept）
- 自定义类型作为容器元素时需确保异常安全
- 移动语义标记为noexcept的类型在容器操作中更高效
- allocator的`allocate`可能抛出`bad_alloc`
