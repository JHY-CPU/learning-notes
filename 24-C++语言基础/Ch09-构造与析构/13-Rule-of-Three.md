# Rule of Three

## 一、概念说明

**Rule of Three**（三法则）是C++98时代的核心原则：**如果一个类需要自定义析构函数、拷贝构造函数或拷贝赋值运算符中的任何一个，那么它很可能需要全部三个**。这是因为这些函数通常都与资源管理相关。

### 1.1 适用场景

当类直接管理动态资源（如裸指针指向的内存）时，必须自定义这三个特殊成员函数，否则编译器生成的默认版本将导致浅拷贝、双重释放等问题。

## 二、具体用法

### 2.1 完整实现示例

```cpp
#include <iostream>
#include <algorithm>
#include <cstring>

class String {
private:
    char* data;
    size_t length;
public:
    // 构造函数
    String(const char* s = "") : length(strlen(s)) {
        data = new char[length + 1];
        strcpy(data, s);
    }

    // ① 析构函数
    ~String() {
        delete[] data;
    }

    // ② 拷贝构造函数
    String(const String& other) : length(other.length) {
        data = new char[length + 1];
        strcpy(data, other.data);
        std::cout << "拷贝构造" << std::endl;
    }

    // ③ 拷贝赋值运算符
    String& operator=(const String& other) {
        if (this != &other) {
            delete[] data;
            length = other.length;
            data = new char[length + 1];
            strcpy(data, other.data);
        }
        std::cout << "拷贝赋值" << std::endl;
        return *this;
    }

    const char* c_str() const { return data; }
    size_t size() const { return length; }
};

int main() {
    String s1("Hello");
    String s2 = s1;           // 拷贝构造
    String s3;
    s3 = s1;                  // 拷贝赋值

    std::cout << "s1: " << s1.c_str() << std::endl;
    std::cout << "s2: " << s2.c_str() << std::endl;
    std::cout << "s3: " << s3.c_str() << std::endl;
    return 0;
}
```

**输出：**
```
拷贝构造
拷贝赋值
s1: Hello
s2: Hello
s3: Hello
```

### 2.2 拷贝交换实现

```cpp
#include <iostream>
#include <algorithm>

class IntBuffer {
private:
    int* ptr;
    size_t sz;
public:
    IntBuffer(size_t s) : ptr(new int[s]()), sz(s) {}
    ~IntBuffer() { delete[] ptr; }

    IntBuffer(const IntBuffer& o) : ptr(new int[o.sz]), sz(o.sz) {
        std::copy(o.ptr, o.ptr + sz, ptr);
    }

    friend void swap(IntBuffer& a, IntBuffer& b) noexcept {
        std::swap(a.ptr, b.ptr);
        std::swap(a.sz, b.sz);
    }

    IntBuffer& operator=(IntBuffer other) {  // 按值传参
        swap(*this, other);
        return *this;
    }
};
```

## 三、注意事项与常见陷阱

- 三法则是C++11之前管理资源的最低要求
- 忘记定义任何一个都可能导致资源泄漏或双重释放
- Rule of Three应被视为Rule of Zero的降级方案
- 在现代C++中，优先考虑使用智能指针和标准容器避免手动资源管理
