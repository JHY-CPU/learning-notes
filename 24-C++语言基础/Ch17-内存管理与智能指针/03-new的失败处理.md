# new的失败处理

## 一、概念说明

当`new`无法分配所需内存时，有两种处理方式：
1. **抛出`std::bad_alloc`异常**（默认行为）
2. **返回`nullptr`**（使用`std::nothrow`）

C++标准库提供了`<new>`头文件，定义了`std::bad_alloc`异常类和`std::nothrow`常量。

```cpp
#include <iostream>
#include <new>

int main() {
    // 方式1：默认抛出异常
    try {
        // 尝试分配巨大内存
        size_t huge = 1024ULL * 1024 * 1024 * 10; // 10GB
        int* p = new int[huge];
        delete[] p;
    } catch (const std::bad_alloc& e) {
        std::cout << "捕获异常: " << e.what() << std::endl;
    }

    // 方式2：使用nothrow，返回nullptr
    int* p2 = new(std::nothrow) int[huge];
    if (p2 == nullptr) {
        std::cout << "分配失败，返回nullptr" << std::endl;
    }
    delete[] p2;

    return 0;
}
```

**输出：**
```
捕获异常: std::bad_alloc
分配失败，返回nullptr
```

## 二、具体用法

### 2.1 设置new_handler

`new_handler`是在`new`失败时调用的回调函数，可用于释放内存或记录日志。

```cpp
#include <iostream>
#include <new>

void myNewHandler() {
    std::cout << "内存分配失败！尝试释放缓存..." << std::endl;
    // 释放一些缓存或记录日志
    throw std::bad_alloc(); // 无法恢复时抛出异常
}

int main() {
    std::set_new_handler(myNewHandler);

    try {
        size_t huge = 1024ULL * 1024 * 1024 * 10;
        int* p = new int[huge];
        delete[] p;
    } catch (const std::bad_alloc& e) {
        std::cout << "最终捕获: " << e.what() << std::endl;
    }

    return 0;
}
```

**输出：**
```
内存分配失败！尝试释放缓存...
最终捕获: std::bad_alloc
```

### 2.2 重载类的operator new

```cpp
#include <iostream>
#include <new>

class MyObject {
public:
    static void* operator new(size_t size) {
        std::cout << "自定义new，大小: " << size << std::endl;
        void* p = std::malloc(size);
        if (!p) throw std::bad_alloc();
        return p;
    }
    static void operator delete(void* p) {
        std::cout << "自定义delete" << std::endl;
        std::free(p);
    }
};

int main() {
    MyObject* obj = new MyObject();
    delete obj;
    return 0;
}
```

**输出：**
```
自定义new，大小: 1
自定义delete
```

## 三、注意事项与常见陷阱

- **默认行为是抛异常**：不要忘记用`try-catch`捕获，或使用`nothrow`。
- **`nothrow`不一定保证不抛异常**：构造函数仍可能抛异常。
- **`new_handler`全局生效**：会影响所有`new`操作，需谨慎设置。
- **重载`new`/`delete`要成对**：类中重载`new`一般也应重载对应的`delete`。
- **`bad_alloc`继承自`exception`**：可以用`std::exception`统一捕获。
