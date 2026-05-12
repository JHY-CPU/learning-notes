# C++17 string_view

## 一、概念说明

`std::string_view`（C++17 §24.1，`<string_view>`）是非拥有（non-owning）的字符串视图，只包含指针和长度，不拷贝数据。它是`std::string`的轻量级替代品，用于函数参数时避免不必要的字符串拷贝。

### 1.1 string_view vs const string&

| 特性 | `const string&` | `string_view` |
|------|----------------|---------------|
| 接受C字符串 | 需要创建临时string | 直接接受 |
| 接受子串 | 需要创建临时string | O(1)切片 |
| 是否拥有数据 | 引用string | 不拥有 |
| 拷贝开销 | 拷贝引用 | 拷贝指针+长度 |

```cpp
#include <iostream>
#include <string_view>
#include <string>

// 接受string_view避免拷贝
std::string_view getExtension(std::string_view filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string_view::npos) return "";
    return filename.substr(pos);
}

int main() {
    // 从各种来源创建string_view
    std::string s = "hello.txt";
    std::string_view sv1(s);           // 从string
    std::string_view sv2("world.cpp"); // 从字面量
    std::string_view sv3 = s;          // 隐式转换

    std::cout << "sv1: " << sv1 << std::endl;
    std::cout << "sv2: " << sv2 << std::endl;
    std::cout << "扩展名: " << getExtension("image.png") << std::endl;

    // 常用操作
    std::string_view text = "Hello, World!";
    std::cout << "substr(7,5): " << text.substr(7, 5) << std::endl;
    std::cout << "find(World): " << text.find("World") << std::endl;

    return 0;
}
```

**输出：**
```
sv1: hello.txt
sv2: world.cpp
扩展名: .png
substr(7,5): World
find(World): 7
```

## 二、具体用法

### 2.1 作为函数参数

```cpp
#include <iostream>
#include <string_view>
#include <string>

// 用string_view替代const string&
void log(std::string_view message) {
    std::cout << "[LOG] " << message << std::endl;
}

// 解析
std::string_view trim(std::string_view sv) {
    auto start = sv.find_first_not_of(" \t\n");
    auto end = sv.find_last_not_of(" \t\n");
    if (start == std::string_view::npos) return "";
    return sv.substr(start, end - start + 1);
}

int main() {
    // 所有这些都不会拷贝
    log("C字符串字面量");
    std::string s = "std::string";
    log(s);
    log(std::string_view("string_view"));

    // trim
    std::string_view padded = "  hello  ";
    std::cout << "trim: '" << trim(padded) << "'" << std::endl;

    return 0;
}
```

**输出：**
```
[LOG] C字符串字面量
[LOG] std::string
[LOG] string_view
trim: 'hello'
```

### 2.2 string_view操作

```cpp
#include <iostream>
#include <string_view>

int main() {
    std::string_view sv = "Hello, World!";

    // 基本操作
    std::cout << "size: " << sv.size() << std::endl;
    std::cout << "empty: " << sv.empty() << std::endl;
    std::cout << "front: " << sv.front() << std::endl;
    std::cout << "back: " << sv.back() << std::endl;
    std::cout << "data: " << sv.data() << std::endl;

    // 子串（O(1)，不拷贝）
    auto sub = sv.substr(7, 5);
    std::cout << "substr: " << sub << std::endl;

    // 查找
    std::cout << "find(World): pos=" << sv.find("World") << std::endl;
    std::cout << "rfind(l): pos=" << sv.rfind('l') << std::endl;
    std::cout << "find_first_of(aeio): pos=" << sv.find_first_of("aeio") << std::endl;

    // 比较
    std::cout << "starts_with(Hello): " << sv.starts_with("Hello") << std::endl;
    std::cout << "ends_with(!): " << sv.ends_with("!") << std::endl;

    // 移除前缀/后缀
    sv.remove_prefix(7);
    std::cout << "remove_prefix: " << sv << std::endl;
    sv.remove_suffix(1);
    std::cout << "remove_suffix: " << sv << std::endl;

    return 0;
}
```

### 2.3 与string转换

```cpp
#include <iostream>
#include <string_view>
#include <string>

int main() {
    // string_view -> string（需要时才拷贝）
    std::string_view sv = "hello";
    std::string s(sv); // 显式转换

    // string -> string_view（零开销）
    std::string str = "world";
    std::string_view sv2 = str; // 隐式转换

    std::cout << "s=" << s << ", sv2=" << sv2 << std::endl;

    // C++20: starts_with / ends_with
    #if __cplusplus >= 202002L
    std::cout << "starts_with: " << sv2.starts_with("wo") << std::endl;
    #endif

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`string_view`不拥有数据**：底层字符串销毁后视图悬垂，是最常见的bug来源。
2. **不能从临时`string`创建`string_view`**：`std::string_view sv = getTempString();`是悬垂引用。
3. **不保证以`\0`结尾**：不能直接传给需要C字符串的函数（如`fopen`）。
4. **`string_view`比`const string&`更通用**：接受C字符串、`string`、字符数组。
5. **修改底层数据会导致`string_view`失效**：`string_view`是只读的。
6. **C++20新增`starts_with`/`ends_with`**：C++17需要手动实现。
7. **`string_view`适合函数参数**：不建议作为类的长期存储成员（除非确定生命周期）。
