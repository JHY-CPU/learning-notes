# arena分配器

## 一、概念说明

Arena（区域）分配器从一个大内存块中顺序分配，释放时**一次性释放整个区域**。适用于需要大量临时分配的场景（如编译器的语法树构建、请求处理）。

优点：分配极快（只需移动指针）、无碎片、批量释放高效。缺点：不能单独释放单个对象。

```cpp
#include <iostream>
#include <vector>
#include <cstddef>

class Arena {
    char* buffer;
    size_t capacity;
    size_t offset;

public:
    Arena(size_t size) : capacity(size), offset(0) {
        buffer = new char[size];
        std::cout << "Arena创建: " << size << " 字节" << std::endl;
    }

    ~Arena() {
        std::cout << "Arena销毁: 使用了" << offset << "/" << capacity << " 字节" << std::endl;
        delete[] buffer;
    }

    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        // 对齐偏移量
        size_t aligned = (offset + alignment - 1) & ~(alignment - 1);
        if (aligned + size > capacity) {
            throw std::bad_alloc();
        }
        void* ptr = buffer + aligned;
        offset = aligned + size;
        return ptr;
    }

    void reset() {
        offset = 0; // 一次性释放所有内存
        std::cout << "Arena重置" << std::endl;
    }

    size_t used() const { return offset; }
};

int main() {
    Arena arena(1024);

    // 在arena上分配对象
    int* a = static_cast<int*>(arena.allocate(sizeof(int)));
    *a = 42;

    double* b = static_cast<double*>(arena.allocate(sizeof(double)));
    *b = 3.14;

    char* c = static_cast<char*>(arena.allocate(32));
    std::strcpy(c, "Hello Arena");

    std::cout << "*a = " << *a << ", *b = " << *b << ", c = " << c << std::endl;
    std::cout << "已使用: " << arena.used() << " 字节" << std::endl;

    // 重置：所有对象的生命周期同时结束
    arena.reset();

    return 0;
}
```

**输出：**
```
Arena创建: 1024 字节
*a = 42, *b = 3.14, c = Hello Arena
已使用: 56 字节
Arena重置
Arena销毁: 使用了0/1024 字节
```

## 二、具体用法

### 2.1 使用std::pmr::monotonic_buffer_resource

```cpp
#include <iostream>
#include <memory_resource>
#include <vector>
#include <string>

int main() {
    char buffer[512];
    std::pmr::monotonic_buffer_resource arena(buffer, sizeof(buffer));

    std::pmr::vector<std::pmr::string> names(&arena);

    names.emplace_back("Alice");
    names.emplace_back("Bob");
    names.emplace_back("Charlie");

    for (const auto& name : names) {
        std::cout << name << std::endl;
    }

    // 所有内存自动释放
    return 0;
}
```

**输出：**
```
Alice
Bob
Charlie
```

### 2.2 请求级Arena

```cpp
#include <iostream>
#include <memory>

class RequestArena {
    static constexpr size_t SIZE = 4096;
    char buffer[SIZE];
    size_t offset = 0;

public:
    template <typename T, typename... Args>
    T* create(Args&&... args) {
        void* ptr = allocate(sizeof(T), alignof(T));
        return ::new (ptr) T(std::forward<Args>(args)...);
    }

    void reset() { offset = 0; }

private:
    void* allocate(size_t size, size_t alignment) {
        size_t aligned = (offset + alignment - 1) & ~(alignment - 1);
        if (aligned + size > SIZE) throw std::bad_alloc();
        void* p = buffer + aligned;
        offset = aligned + size;
        return p;
    }
};

struct Request {
    int id;
    std::string path;
    Request(int i, std::string p) : id(i), path(std::move(p)) {}
};

void handleRequest(int id, const std::string& path) {
    RequestArena arena; // 每个请求一个arena
    auto* req = arena.create<Request>(id, path);
    std::cout << "处理请求 #" << req->id << ": " << req->path << std::endl;
    // 请求结束，arena自动析构，无需逐个释放
}

int main() {
    handleRequest(1, "/api/users");
    handleRequest(2, "/api/orders");
    return 0;
}
```

**输出：**
```
处理请求 #1: /api/users
处理请求 #2: /api/orders
```

## 三、注意事项与常见陷阱

- **不能单独释放对象**：只能批量重置。
- **对象析构需手动调用**：`arena.reset()`前应析构所有对象。
- **栈上Arena大小有限**：大对象应使用堆上分配的Arena。
- **线程不安全**：多线程使用需加锁或每线程一个Arena。
- **适合生命周期一致的对象组**：如一次HTTP请求中的所有临时对象。
