# swap与引用

## 一、概念说明

`swap`是交换两个变量值的经典操作，实现依赖引用。`std::swap`是标准库提供的通用交换函数，对自定义类型可通过特化或ADL（参数依赖查找）提供优化版本。

## 二、具体用法

### 2.1 基本swap实现

```cpp
void mySwap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int x = 10, y = 20;
mySwap(x, y);
std::cout << x << " " << y << std::endl;  // 输出: 20 10
```

### 2.2 std::swap

```cpp
#include <utility>

int a = 1, b = 2;
std::swap(a, b);
std::cout << a << " " << b << std::endl;  // 输出: 2 1

std::string s1 = "Hello", s2 = "World";
std::swap(s1, s2);
std::cout << s1 << " " << s2 << std::endl;  // 输出: World Hello
```

### 2.3 自定义类型的swap

```cpp
class Buffer {
    int* data;
    size_t size;
public:
    Buffer(size_t n) : data(new int[n]), size(n) {}

    // 成员swap
    void swap(Buffer& other) noexcept {
        std::swap(data, other.data);
        std::swap(size, other.size);
    }

    // 友元swap（ADL查找）
    friend void swap(Buffer& a, Buffer& b) noexcept {
        a.swap(b);
    }

    ~Buffer() { delete[] data; }
};

Buffer b1(100), b2(200);
swap(b1, b2);  // ADL找到Buffer的swap
```

### 2.4 swap实现移动赋值

```cpp
class Resource {
    int* ptr;
public:
    // copy-and-swap惯用法
    Resource& operator=(Resource other) {  // 按值接收（触发拷贝/移动）
        swap(other);  // 交换内容
        return *this;  // other析构时释放旧资源
    }
    void swap(Resource& other) noexcept {
        std::swap(ptr, other.ptr);
    }
};
```

### 2.5 noexcept的重要性

```cpp
// swap通常应标记noexcept
void swap(MyType& a, MyType& b) noexcept {
    using std::swap;
    swap(a.data, b.data);  // 对内置类型swap是noexcept
}
```

## 三、注意事项与常见陷阱

- `std::swap`对大型对象效率低（三次拷贝），自定义swap使用移动
- ADL允许编译器自动找到类型的swap函数
- 使用`using std::swap; swap(a, b);`的模式启用ADL
- `swap`应标记`noexcept`（vector扩容等场景需要）
- copy-and-swap是实现异常安全赋值的经典模式
