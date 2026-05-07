# allocator_traits

## 一、概念说明

`std::allocator_traits`提供了统一的allocator操作接口，即使自定义allocator缺少某些方法，也能通过默认实现正常工作。它是STL容器与allocator之间的适配层。

使用`allocator_traits`而非直接调用allocator方法，使得代码更加通用和兼容。

```cpp
#include <iostream>
#include <memory>

// 一个最简自定义allocator（缺少construct等方法）
template <typename T>
struct MinimalAllocator {
    using value_type = T;
    MinimalAllocator() = default;

    template <typename U>
    MinimalAllocator(const MinimalAllocator<U>&) {}

    T* allocate(size_t n) {
        std::cout << "MinimalAllocator::allocate(" << n << ")" << std::endl;
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, size_t n) {
        std::cout << "MinimalAllocator::deallocate(" << n << ")" << std::endl;
        ::operator delete(p);
    }
};

int main() {
    using Traits = std::allocator_traits<MinimalAllocator<int>>;
    MinimalAllocator<int> alloc;

    // 使用traits的统一接口（即使allocator没有construct方法）
    int* p = Traits::allocate(alloc, 3);
    Traits::construct(alloc, &p[0], 10);
    Traits::construct(alloc, &p[1], 20);
    Traits::construct(alloc, &p[2], 30);

    std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;

    Traits::destroy(alloc, &p[0]);
    Traits::destroy(alloc, &p[1]);
    Traits::destroy(alloc, &p[2]);
    Traits::deallocate(alloc, p, 3);

    return 0;
}
```

**输出：**
```
MinimalAllocator::allocate(3)
10 20 30
MinimalAllocator::deallocate(3)
```

## 二、具体用法

### 2.1 allocator_traits提供的默认实现

```cpp
#include <iostream>
#include <memory>

template <typename T>
struct MyAllocator {
    using value_type = T;

    T* allocate(size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }
    void deallocate(T* p, size_t) {
        ::operator delete(p);
    }
    // 没有construct、destroy、max_size等方法
};

int main() {
    using Traits = std::allocator_traits<MyAllocator<int>>;

    // traits提供默认实现
    std::cout << "max_size: " << Traits::max_size(MyAllocator<int>{}) << std::endl;

    MyAllocator<int> alloc;
    int* p = Traits::allocate(alloc, 1);

    // construct使用placement new的默认实现
    Traits::construct(alloc, p, 42);
    std::cout << "值: " << *p << std::endl;

    Traits::destroy(alloc, p);
    Traits::deallocate(alloc, p, 1);

    return 0;
}
```

**输出：**
```
max_size: 4611686018427387903
值: 42
```

### 2.2 rebind机制

```cpp
#include <iostream>
#include <memory>

template <typename T>
struct MyAlloc {
    using value_type = T;
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }
    void deallocate(T* p, size_t) { ::operator delete(p); }
};

int main() {
    // 用allocator_traits获取rebind后的类型
    using IntAlloc = MyAlloc<int>;
    using CharAlloc = std::allocator_traits<IntAlloc>::rebind_alloc<char>;

    CharAlloc charAlloc;
    char* p = charAlloc.allocate(10);
    std::cout << "rebind后分配了10个char" << std::endl;
    charAlloc.deallocate(p, 10);

    return 0;
}
```

**输出：**
```
rebind后分配了10个char
```

## 三、注意事项与常见陷阱

- **始终通过`allocator_traits`操作allocator**，而非直接调用allocator方法。
- **`construct`在C++20中从allocator移除**，必须通过traits使用。
- **自定义allocator只需实现最少接口**（`value_type`、`allocate`、`deallocate`），traits补充其余。
- **`rebind_alloc`用于在allocator中获取其他类型的分配器**（如链表节点）。
- **`propagate_on_container_copy_assignment`等标签**控制容器操作时allocator的传播行为。
