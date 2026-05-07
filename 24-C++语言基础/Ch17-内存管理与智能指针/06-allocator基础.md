# allocator基础

## 一、概念说明

`std::allocator`是STL容器默认使用的内存分配器，将**内存分配**与**对象构造**分离。它提供了`allocate`（分配原始内存）和`deallocate`（释放内存）方法，以及`construct`和`destroy`方法。

这种分离使得STL容器可以高效管理内存，避免不必要的构造/析构调用。

```cpp
#include <iostream>
#include <memory>
#include <string>

int main() {
    std::allocator<std::string> alloc;

    // 分配3个string的原始内存（未构造）
    std::string* arr = alloc.allocate(3);

    // 在分配的内存上构造对象
    alloc.construct(&arr[0], "Hello");
    alloc.construct(&arr[1], "World");
    alloc.construct(&arr[2], "C++");

    for (int i = 0; i < 3; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // 析构对象（不释放内存）
    for (int i = 0; i < 3; ++i) {
        alloc.destroy(&arr[i]);
    }

    // 释放原始内存
    alloc.deallocate(arr, 3);

    return 0;
}
```

**输出：**
```
Hello World C++
```

## 二、具体用法

### 2.1 allocator的基本操作

```cpp
#include <iostream>
#include <memory>

int main() {
    std::allocator<int> alloc;

    // 分配5个int的空间
    int* p = alloc.allocate(5);
    std::cout << "分配了5个int的空间" << std::endl;

    // 值初始化
    for (int i = 0; i < 5; ++i) {
        alloc.construct(&p[i], i * 10);
    }

    std::cout << "元素: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << p[i] << " ";
    }
    std::cout << std::endl;

    // 析构并释放
    for (int i = 0; i < 5; ++i) {
        alloc.destroy(&p[i]);
    }
    alloc.deallocate(p, 5);

    return 0;
}
```

**输出：**
```
分配了5个int的空间
元素: 0 10 20 30 40
```

### 2.2 用allocator实现简单容器

```cpp
#include <iostream>
#include <memory>

template <typename T>
class SimpleVector {
    std::allocator<T> alloc;
    T* data;
    size_t size_;
    size_t capacity_;
public:
    SimpleVector(size_t cap = 4)
        : alloc(), data(alloc.allocate(cap)), size_(0), capacity_(cap) {}

    ~SimpleVector() {
        for (size_t i = 0; i < size_; ++i)
            alloc.destroy(&data[i]);
        alloc.deallocate(data, capacity_);
    }

    void push_back(const T& val) {
        if (size_ < capacity_)
            alloc.construct(&data[size_++], val);
    }

    T& operator[](size_t i) { return data[i]; }
    size_t size() const { return size_; }
};

int main() {
    SimpleVector<int> vec;
    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);

    for (size_t i = 0; i < vec.size(); ++i)
        std::cout << vec[i] << " ";
    std::cout << std::endl;

    return 0;
}
```

**输出：**
```
10 20 30
```

## 三、注意事项与常见陷阱

- **`allocate`只分配内存不构造**：必须配合`construct`使用。
- **`deallocate`只释放内存不析构**：必须先`destroy`再`deallocate`。
- **`construct`/`destroy`在C++17中已弃用，C++20中移除**：应使用`allocator_traits`或`std::construct_at`。
- **分配数量必须匹配**：`deallocate(p, n)`中的`n`应与`allocate(n)`一致。
- **allocator不处理异常**：构造失败时需手动回滚。
