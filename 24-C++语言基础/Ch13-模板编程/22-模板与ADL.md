# 模板与ADL

## 一、概念说明

ADL（Argument-Dependent Lookup，参数依赖查找），也称Koenig查找，是C++中查找非限定函数名的规则。当调用一个没有命名空间前缀的函数时，编译器除了查找当前作用域，还会根据实参的类型到其关联的命名空间中查找。这在模板编程中至关重要。

## 二、具体用法

### 2.1 ADL基本原理

```cpp
namespace MyLib {
    struct Widget {};

    // 在Widget的命名空间中定义函数
    void process(const Widget&) {
        std::cout << "MyLib::process(Widget)" << std::endl;
    }

    void print(const Widget&) {
        std::cout << "MyLib::print(Widget)" << std::endl;
    }
}

int main() {
    MyLib::Widget w;
    // 没有用 MyLib::process，但ADL会找到它
    process(w);  // MyLib::process(Widget) ✓
    print(w);    // MyLib::print(Widget) ✓
}
```

### 2.2 模板中的ADL

```cpp
namespace Math {
    template <typename T>
    struct Complex {
        T real, imag;
    };

    // 在Math命名空间定义
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Complex<T>& c) {
        return os << c.real << "+" << c.imag << "i";
    }
}

// 模板函数中使用ADL
template <typename T>
void display(const T& val) {
    // operator<< 通过ADL找到Math::operator<<
    std::cout << val << std::endl;
}

int main() {
    Math::Complex<double> c{3.0, 4.0};
    display(c);  // 3+4i （通过ADL找到operator<<）
}
```

### 2.3 swap惯用法

```cpp
// 自定义swap的正确写法
namespace MyLib {
    class Buffer {
        int* data;
        std::size_t size;
    public:
        Buffer(std::size_t n) : data(new int[n]), size(n) {}
        ~Buffer() { delete[] data; }

        // 成员swap
        void swap(Buffer& other) noexcept {
            std::swap(data, other.data);
            std::swap(size, other.size);
        }

        friend void swap(Buffer& a, Buffer& b) noexcept { a.swap(b); }
    };
}

template <typename T>
void safe_swap(T& a, T& b) {
    // 用using声明引入std::swap，然后让ADL找到更好的版本
    using std::swap;
    swap(a, b);  // ADL优先找到MyLib::swap，否则用std::swap
}
```

### 2.4 ADL的限制

```cpp
namespace NS {
    struct S {};
    void func(S) { std::cout << "NS::func" << std::endl; }
}

void func(NS::S) { std::cout << "::func" << std::endl; }

int main() {
    NS::S s;
    func(s);  // ::func（当前作用域优先于ADL）
    // 需要显式调用 NS::func(s) 才能调用命名空间版本
}
```

## 三、注意事项与常见陷阱

- ADL只查找参数类型的"关联命名空间"，不是所有命名空间
- 关联命名空间包括：类型定义的命名空间、模板参数类型的命名空间等
- 当前作用域的函数优先于ADL找到的函数
- ADL不适用于有显式命名空间限定的调用（如`std::swap(a,b)`不会触发ADL）
- swap惯用法：`using std::swap; swap(a, b);`是标准做法
- ADL可能导致意外的函数匹配，尤其在嵌套命名空间中
- `friend`函数定义在类内时，只能通过ADL找到
