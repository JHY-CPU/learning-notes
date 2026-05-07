# std::string_view

## 一、概念说明

`std::string_view`（C++17）是一个**非拥有**（non-owning）的字符串引用，提供对字符串数据的只读访问。它不管理内存，仅保存指针和长度，因此创建和传递的**开销极小**。

`string_view`适用于函数参数传递，避免不必要的字符串拷贝。

## 二、具体用法

### 2.1 基本使用

```cpp
#include <string_view>

// 从字面量构造（无拷贝）
std::string_view sv1 = "Hello, World!";
std::cout << sv1 << std::endl;  // 输出: Hello, World!

// 从std::string构造（无拷贝）
std::string str = "C++ Programming";
std::string_view sv2 = str;
std::cout << sv2 << std::endl;  // 输出: C++ Programming

// 从C字符串构造
const char* cstr = "Hello";
std::string_view sv3 = cstr;
```

### 2.2 常用操作

```cpp
std::string_view sv = "Hello World";

std::cout << sv.size() << std::endl;     // 输出: 11
std::cout << sv.substr(0, 5) << std::endl;  // 输出: Hello
std::cout << sv.find("World") << std::endl; // 输出: 6
std::cout << sv[0] << std::endl;           // 输出: H

// 前缀/后缀移除（C++20）
// sv.remove_prefix(6);  // "World"
// sv.remove_suffix(1);  // "Worl"
```

### 2.3 作为函数参数（推荐用法）

```cpp
// string_view可接受string、字面量、C字符串，且无拷贝
void printName(std::string_view name) {
    std::cout << "Name: " << name << std::endl;
}

std::string s = "Alice";
printName(s);           // 从string构造
printName("Bob");        // 从字面量构造
printName(s.c_str());   // 从C字符串构造
// 输出:
// Name: Alice
// Name: Bob
// Name: Alice
```

### 2.4 零拷贝子串

```cpp
std::string_view sv = "Hello World";

// substr返回新的string_view，不拷贝数据
auto sub = sv.substr(6, 5);  // "World"
std::cout << sub << std::endl;  // 输出: World
```

## 三、注意事项与常见陷阱

- **string_view不拥有数据**，底层数据被释放后使用是悬垂引用
- 不要返回局部`string`的`string_view`
- `string_view`不能修改底层数据（只读）
- `string_view`不保证以`\0`结尾，使用`data()`时注意
- 需要长期持有或修改时，转换为`std::string`
