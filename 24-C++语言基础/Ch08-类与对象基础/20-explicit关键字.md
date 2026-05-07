# explicit关键字

## 一、概念说明

`explicit`关键字用于**防止隐式转换**。它可以修饰构造函数（C++11起）和类型转换运算符（C++11起）。隐式转换虽然方便但可能导致意外行为，`explicit`让转换必须显式进行。

## 二、具体用法

### 2.1 explicit构造函数

```cpp
class Wrapper {
    int value;
public:
    explicit Wrapper(int v) : value(v) {}
    int get() const { return value; }
};

void process(Wrapper w) {
    std::cout << w.get() << std::endl;
}

// Wrapper w = 42;  // 编译错误：不能隐式转换
Wrapper w(42);       // OK：显式构造
// process(42);      // 编译错误：不能隐式转换
process(Wrapper(42));  // OK：显式转换
// 输出: 42
```

### 2.2 explicit类型转换运算符

```cpp
class BoolLike {
    bool value;
public:
    BoolLike(bool v) : value(v) {}

    explicit operator bool() const {
        return value;
    }
};

BoolLike b(true);

// if (b) { }  // C++11起OK：上下文转换（if/while/for等）

// bool x = b;  // 编译错误：不能隐式转换
bool x = static_cast<bool>(b);  // OK：显式转换
std::cout << x << std::endl;  // 输出: 1
```

### 2.3 标准库中的应用

```cpp
// std::string的构造函数
std::string s = "Hello";  // OK：非explicit，允许隐式

// std::vector的构造函数
// std::vector<int> v = 10;  // 编译错误：explicit(size_t)
std::vector<int> v(10);      // OK：显式调用

// std::unique_ptr
// std::unique_ptr<int> p = new int(42);  // 编译错误：explicit
std::unique_ptr<int> p(new int(42));      // OK
```

### 2.4 对比有无explicit

```cpp
class Safe {
    int val;
public:
    explicit Safe(int v) : val(v) {}
};

class Unsafe {
    int val;
public:
    Unsafe(int v) : val(v) {}  // 允许隐式转换
};

// Unsafe可隐式转换
void useUnsafe(Unsafe u) {}
useUnsafe(42);  // 隐式调用Unsafe(42) - 可能非预期

// Safe必须显式
void useSafe(Safe s) {}
// useSafe(42);  // 编译错误
useSafe(Safe(42));  // OK
```

## 三、注意事项与常见陷阱

- 单参数构造函数默认允许隐式转换，`explicit`可阻止
- `explicit`不影响直接初始化：`ClassName obj(args)`
- C++11起`explicit`可用于转换运算符
- STL中很多构造函数是`explicit`的（防止意外转换）
- Rule of Thumb：单参数构造函数通常应声明为`explicit`
