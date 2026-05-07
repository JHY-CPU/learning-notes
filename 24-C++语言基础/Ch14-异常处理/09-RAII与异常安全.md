# RAII与异常安全

## 一、概念说明

RAII（Resource Acquisition Is Initialization，资源获取即初始化）是C++最核心的资源管理模式。通过在构造函数中获取资源、在析构函数中释放资源，确保即使发生异常，资源也能被正确释放。

## 二、具体用法

### 2.1 基本RAII模式

```cpp
#include <iostream>
#include <fstream>
#include <stdexcept>

class FileHandler {
    std::FILE* file;
public:
    FileHandler(const char* filename, const char* mode) {
        file = std::fopen(filename, mode);
        if (!file) {
            throw std::runtime_error("无法打开文件");
        }
    }

    ~FileHandler() {
        if (file) {
            std::fclose(file);
            std::cout << "文件已关闭" << std::endl;
        }
    }

    // 禁止拷贝
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;

    std::FILE* get() const { return file; }
};

void write_file() {
    FileHandler fh("test.txt", "w");
    std::fputs("Hello RAII\n", fh.get());
    throw std::runtime_error("写入中发生异常");
    // fh的析构函数仍会被调用，文件自动关闭
}

int main() {
    try {
        write_file();
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
    }
    // 输出: 文件已关闭
}
```

### 2.2 使用智能指针

```cpp
#include <memory>

class Connection {
    std::string name;
public:
    Connection(const std::string& n) : name(n) {
        std::cout << "连接: " << name << std::endl;
    }
    ~Connection() {
        std::cout << "断开: " << name << std::endl;
    }
    void query() { std::cout << "查询: " << name << std::endl; }
};

void process() {
    auto conn = std::make_unique<Connection>("数据库");
    conn->query();
    throw std::runtime_error("查询失败");
    // conn自动释放，Connection被析构
}

int main() {
    try {
        process();
    } catch (...) {
        std::cout << "异常已处理" << std::endl;
    }
}
```

### 2.3 RAII守卫类

```cpp
template <typename T>
class ScopeGuard {
    T cleanup_func;
    bool active;
public:
    explicit ScopeGuard(T func) : cleanup_func(std::move(func)), active(true) {}

    ~ScopeGuard() {
        if (active) cleanup_func();
    }

    void dismiss() { active = false; }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
};

// C++14 便捷函数
template <typename T>
ScopeGuard<T> make_guard(T func) {
    return ScopeGuard<T>(std::move(func));
}

void demo_guard() {
    int* data = new int[100];
    auto guard = make_guard([&]() {
        delete[] data;
        std::cout << "内存已释放" << std::endl;
    });

    // 操作data...
    throw std::runtime_error("操作失败");
    // guard析构时自动释放内存
}
```

### 2.4 C++标准库RAII工具

```cpp
#include <mutex>

std::mutex mtx;

void thread_safe_operation() {
    std::lock_guard<std::mutex> lock(mtx);  // RAII锁
    // 临界区代码
    // 即使抛异常，lock析构时也会释放锁
    if (true) throw std::runtime_error("线程错误");
}
```

## 三、注意事项与常见陷阱

- RAII是C++资源管理的核心范式，应始终使用
- 智能指针（`unique_ptr`、`shared_ptr`）是最常用的RAII工具
- `lock_guard`、`unique_lock`提供锁的RAII管理
- 自定义RAII类需要正确处理拷贝/移动语义
- RAII对象应在栈上分配（非堆上），确保析构时机正确
- 避免在析构函数中抛异常
