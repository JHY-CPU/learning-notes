# default与delete

## 一、概念说明

C++11引入`= default`和`= delete`，用于显式控制特殊成员函数的生成。`= default`请求编译器生成默认实现，`= delete`禁止函数被调用。

## 二、具体用法

### 2.1 = default

```cpp
class Data {
    int value;
public:
    Data() = default;  // 显式要求编译器生成默认构造
    Data(int v) : value(v) {}

    // 显式要求默认拷贝/移动
    Data(const Data&) = default;
    Data& operator=(const Data&) = default;
    Data(Data&&) = default;
    Data& operator=(Data&&) = default;

    ~Data() = default;
};

Data d1;       // 使用默认构造
Data d2(42);   // 使用参数化构造
Data d3 = d2;  // 使用默认拷贝构造
```

### 2.2 = delete

```cpp
class NonCopyable {
public:
    NonCopyable() = default;

    // 禁止拷贝
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;

    // 允许移动
    NonCopyable(NonCopyable&&) = default;
    NonCopyable& operator=(NonCopyable&&) = default;
};

NonCopyable a;
// NonCopyable b = a;  // 编译错误：拷贝已delete
NonCopyable b = std::move(a);  // OK：移动允许
```

### 2.3 delete阻止特定转换

```cpp
class IntOnly {
public:
    IntOnly(int) {}
    IntOnly(double) = delete;  // 禁止double转换
    IntOnly(const char*) = delete;  // 禁止字符串转换
};

IntOnly a(42);      // OK
// IntOnly b(3.14);  // 编译错误
// IntOnly c("hi");  // 编译错误
```

### 2.4 Rule of Zero

```cpp
// 最佳实践：不管理资源时，不定义任何特殊成员函数
class Simple {
    std::string name;
    std::vector<int> data;
    // 编译器自动生成所有正确的特殊成员函数
    // （string和vector已正确管理资源）
};
```

### 2.5 Rule of Five

```cpp
// 管理资源时：五个特殊成员函数一起考虑
class Resource {
    int* ptr;
public:
    Resource(int v) : ptr(new int(v)) {}
    ~Resource() { delete ptr; }

    Resource(const Resource& o) : ptr(new int(*o.ptr)) {}
    Resource& operator=(const Resource& o) {
        if (this != &o) { delete ptr; ptr = new int(*o.ptr); }
        return *this;
    }
    Resource(Resource&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    Resource& operator=(Resource&& o) noexcept {
        if (this != &o) { delete ptr; ptr = o.ptr; o.ptr = nullptr; }
        return *this;
    }
};
```

## 三、注意事项与常见陷阱

- `= default`只能用于特殊成员函数
- `= delete`可用于任何函数（包括非成员函数）
- `unique_ptr`通过`= delete`拷贝操作实现独占所有权
- Rule of Zero：如果可以不管理资源，就不要定义任何特殊成员函数
- 删除的函数在重载决议中仍可见，选择它会导致编译错误
