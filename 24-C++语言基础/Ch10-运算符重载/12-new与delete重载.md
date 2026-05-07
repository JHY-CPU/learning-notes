# new与delete重载

## 一、概念说明

`operator new`和`operator delete`可以被重载，用于自定义内存分配策略，如内存池、对齐分配、内存追踪等。它们可以是全局的或类特定的。

## 二、具体用法

### 2.1 类特定的new/delete

```cpp
#include <iostream>
#include <cstdlib>

class Tracked {
public:
    static int allocCount;
    static int deallocCount;

    // 重载operator new
    static void* operator new(size_t size) {
        ++allocCount;
        std::cout << "[分配] " << size << " 字节" << std::endl;
        return ::operator new(size);
    }

    // 重载operator delete
    static void operator delete(void* ptr) {
        ++deallocCount;
        std::cout << "[释放]" << std::endl;
        ::operator delete(ptr);
    }

    // 数组版本
    static void* operator new[](size_t size) {
        ++allocCount;
        return ::operator new[](size);
    }
    static void operator delete[](void* ptr) {
        ++deallocCount;
        ::operator delete[](ptr);
    }
};

int Tracked::allocCount = 0;
int Tracked::deallocCount = 0;

int main() {
    Tracked* p = new Tracked();
    delete p;
    Tracked* arr = new Tracked[3];
    delete[] arr;
    std::cout << "分配次数: " << Tracked::allocCount
              << ", 释放次数: " << Tracked::deallocCount << std::endl;
    return 0;
}
```

**输出：**
```
[分配] 1 字节
[释放]
[分配] 13 字节
[释放]
分配次数: 2, 释放次数: 2
```

### 2.2 对齐版本（C++17）

```cpp
#include <iostream>
#include <new>

class Aligned {
    alignas(64) char data[64];
public:
    static void* operator new(size_t size) {
        return ::operator new(size, std::align_val_t{64});
    }
    static void operator delete(void* ptr) {
        ::operator delete(ptr, std::align_val_t{64});
    }
};
```

## 三、注意事项与常见陷阱

- 自定义`new`必须返回有效指针或抛出`std::bad_alloc`
- `operator new[]`和`operator delete[]`必须配对使用
- 重载不会影响全局`::operator new`的行为
- placement new不需要也不能被替换
- 注意对齐要求（使用`std::align_val_t`）
