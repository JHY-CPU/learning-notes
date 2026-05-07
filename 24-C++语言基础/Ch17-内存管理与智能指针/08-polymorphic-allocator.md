# polymorphic allocator (PMR)

## 一、概念说明

C++17引入的`std::pmr::polymorphic_allocator`允许在运行时选择不同的内存资源（memory resource），无需重新编译模板。它解决了传统allocator的模板化导致的类型不兼容问题。

内存资源（`memory_resource`）是PMR的核心抽象，定义了`do_allocate`/`do_deallocate`/`do_is_equal`三个虚函数。

```cpp
#include <iostream>
#include <memory_resource>
#include <vector>

int main() {
    // 使用单调缓冲区资源（极快，适合临时分配）
    char buffer[1024];
    std::pmr::monotonic_buffer_resource pool{buffer, sizeof(buffer)};

    // 用PMR分配器的vector
    std::pmr::vector<int> vec(&pool);
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    for (int v : vec)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "使用了 " << sizeof(buffer) << " 字节的栈缓冲区" << std::endl;

    return 0;
}
```

**输出：**
```
1 2 3
使用了 1024 字节的栈缓冲区
```

## 二、具体用法

### 2.1 不同内存资源

```cpp
#include <iostream>
#include <memory_resource>
#include <vector>

int main() {
    // 1. 默认资源（使用全局new/delete）
    std::pmr::vector<int> v1;
    v1.push_back(10);
    std::cout << "v1 使用默认资源" << std::endl;

    // 2. 同步池资源（线程安全的内存池）
    std::pmr::synchronized_pool_resource syncPool;
    std::pmr::vector<int> v2(&syncPool);
    v2.push_back(20);
    std::cout << "v2 使用同步池资源" << std::endl;

    // 3. 单调缓冲区（一次性释放，最快）
    char buf[512];
    std::pmr::monotonic_buffer_resource monoPool(buf, sizeof(buf));
    std::pmr::vector<int> v3(&monoPool);
    v3.push_back(30);
    std::cout << "v3 使用单调缓冲区资源" << std::endl;

    std::cout << "v1[0]=" << v1[0] << " v2[0]=" << v2[0] << " v3[0]=" << v3[0] << std::endl;

    return 0;
}
```

**输出：**
```
v1 使用默认资源
v2 使用同步池资源
v3 使用单调缓冲区资源
v1[0]=10 v2[0]=20 v3[0]=30
```

### 2.2 自定义内存资源

```cpp
#include <iostream>
#include <memory_resource>
#include <vector>

class LoggingResource : public std::pmr::memory_resource {
    std::pmr::memory_resource* upstream_;
protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        std::cout << "分配 " << bytes << " 字节" << std::endl;
        return upstream_->allocate(bytes, alignment);
    }
    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
        std::cout << "释放 " << bytes << " 字节" << std::endl;
        upstream_->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }
public:
    LoggingResource() : upstream_(std::pmr::get_default_resource()) {}
};

int main() {
    LoggingResource logger;
    std::pmr::vector<int> vec(&logger);

    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    return 0;
}
```

**输出：**
```
分配 16 字节
分配 32 字节
释放 16 字节
```

## 三、注意事项与常见陷阱

- **PMR容器是独立类型**：`pmr::vector<int>`与`vector<int>`类型不同，不能直接赋值。
- **单调缓冲区资源析构时释放所有内存**：不能在其中析构部分对象。
- **`synchronized_pool_resource`线程安全但有开销**：单线程应使用`unsynchronized_pool_resource`。
- **C++20中`std::pmr`从`<memory_resource>`移入`<memory>`**：注意头文件变化。
- **PMR解决了allocator模板化的类型问题**：不同allocator的容器可以互操作。
