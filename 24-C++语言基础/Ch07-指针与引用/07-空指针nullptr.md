# 空指针 nullptr

## 一、概念说明

`nullptr`是C++11引入的空指针字面量，类型为`std::nullptr_t`。它替代了传统的`NULL`（宏，通常定义为0）和整数0，提供了**类型安全**的空指针表示。

`nullptr`可以隐式转换为任意指针类型，但不能转换为整数类型（除`bool`）。

## 二、具体用法

### 2.1 nullptr vs NULL vs 0

```cpp
int* p1 = nullptr;  // C++11推荐
int* p2 = NULL;     // 宏，等于0或((void*)0)
int* p3 = 0;        // 字面量0

// nullptr类型安全
void func(int)    { std::cout << "int版本\n"; }
void func(int*)   { std::cout << "int*版本\n"; }

func(0);        // 输出: int版本（不是预期！）
func(NULL);     // 输出: int版本（同上，取决于实现）
func(nullptr);  // 输出: int*版本（正确！）
```

### 2.2 检查空指针

```cpp
int* ptr = nullptr;

if (ptr == nullptr) {
    std::cout << "空指针\n";
}

if (!ptr) {
    std::cout << "也是空指针\n";
}
// 输出:
// 空指针
// 也是空指针
```

### 2.3 std::nullptr_t

```cpp
// nullptr有独立类型：std::nullptr_t
std::cout << typeid(nullptr).name() << std::endl;
// 输出: decltype(nullptr) (或类似)

// 可用于模板
template <typename T>
void check(T value) {
    if constexpr (std::is_same_v<T, std::nullptr_t>) {
        std::cout << "是nullptr\n";
    }
}
check(nullptr);  // 输出: 是nullptr
```

### 2.4 安全使用模式

```cpp
class Resource {
    FILE* file;
public:
    Resource() : file(nullptr) {}

    bool open(const char* name) {
        file = fopen(name, "r");
        return file != nullptr;
    }

    ~Resource() {
        if (file) fclose(file);
    }
};
```

## 三、注意事项与常见陷阱

- 始终使用`nullptr`，不要使用`NULL`或`0`表示空指针
- `nullptr`不能与整数比较（除了`bool`转换）
- `sizeof(nullptr)`等于`sizeof(void*)`（通常4或8字节）
- 重载决议中`nullptr`优先匹配指针类型而非整数类型
- `nullptr`的引入解决了C++中`f(0)`的歧义问题
