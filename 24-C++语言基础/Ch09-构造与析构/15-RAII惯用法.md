# RAII惯用法

## 一、概念说明

**RAII**（Resource Acquisition Is Initialization，资源获取即初始化）是C++最核心的资源管理范式。其核心思想是：**在构造函数中获取资源，在析构函数中释放资源**。利用C++对象的自动生命周期管理，确保资源在任何情况下（包括异常）都能被正确释放。

## 二、具体用法

### 2.1 文件句柄RAII

```cpp
#include <iostream>
#include <fstream>
#include <string>

class FileGuard {
private:
    std::fstream file;
public:
    FileGuard(const std::string& filename, std::ios::openmode mode)
        : file(filename, mode) {
        if (!file.is_open())
            throw std::runtime_error("无法打开文件: " + filename);
    }

    ~FileGuard() {
        if (file.is_open()) {
            file.close();
            std::cout << "文件已安全关闭" << std::endl;
        }
    }

    std::fstream& get() { return file; }

    // 禁止拷贝
    FileGuard(const FileGuard&) = delete;
    FileGuard& operator=(const FileGuard&) = delete;
};

int main() {
    try {
        FileGuard fg("test.txt", std::ios::out);
        fg.get() << "Hello RAII!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    // 文件在此处自动关闭，即使发生异常也会关闭
    return 0;
}
```

**输出：**
```
文件已安全关闭
```

### 2.2 锁守卫（Lock Guard）

```cpp
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;
int shared_counter = 0;

void increment(int times) {
    for (int i = 0; i < times; ++i) {
        std::lock_guard<std::mutex> lock(mtx);  // RAII锁
        ++shared_counter;
        // 离开作用域自动解锁，即使抛异常也不会死锁
    }
}

int main() {
    std::thread t1(increment, 10000);
    std::thread t2(increment, 10000);
    t1.join(); t2.join();
    std::cout << "counter = " << shared_counter << std::endl;
    return 0;
}
```

**输出：**
```
counter = 20000
```

### 2.3 自定义作用域守卫

```cpp
#include <iostream>
#include <functional>

class ScopeGuard {
    std::function<void()> cleanup;
    bool dismissed = false;
public:
    explicit ScopeGuard(std::function<void()> fn) : cleanup(std::move(fn)) {}

    ~ScopeGuard() {
        if (!dismissed) cleanup();
    }

    void dismiss() { dismissed = true; }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
};

int main() {
    int* arr = new int[100];
    ScopeGuard guard([&]() { delete[] arr; std::cout << "内存已释放" << std::endl; });

    // 使用arr做操作...
    arr[0] = 42;

    // 无论后续发生什么，arr都会被释放
    return 0;
}
```

**输出：**
```
内存已释放
```

## 三、注意事项与常见陷阱

- RAII保证异常安全，是C++异常处理的基石
- 标准库已提供`std::lock_guard`、`std::unique_ptr`等RAII工具
- 自定义RAII类应禁止或正确实现拷贝/移动操作
- RAII对象应遵循"单一职责"原则，每个对象只管理一种资源
- `std::unique_ptr`的自定义删除器可管理任意资源（不止内存）
