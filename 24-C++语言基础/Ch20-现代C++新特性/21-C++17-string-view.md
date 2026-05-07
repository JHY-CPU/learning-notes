# C++17 string_view

## 一、概念说明

`std::string_view`（`<string_view>`）是非拥有（non-owning）的字符串视图，只包含指针和长度，不拷贝数据。用于函数参数时避免不必要的`std::string`拷贝。

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
    std::cout << "starts_with(Hello): " << text.starts_with("Hello") << std::endl;

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
starts_with(Hello): 1
```

## 二、具体用法

### 2.1 作为函数参数

```cpp
#include <iostream>
#include <string_view>

// 用string_view替代const string&
void log(std::string_view message) {
    std::cout << "[LOG] " << message << std::endl;
}

// string_view + string字面量
void print(std::string_view sv) {
    std::cout << "长度=" << sv.size() << ", 内容=" << sv << std::endl;
}

int main() {
    // 所有这些都不会拷贝
    log("C字符串字面量");
    std::string s = "std::string";
    log(s);
    log(std::string_view("string_view"));

    // 注意：string_view不拥有数据
    // 危险：悬垂视图
    // std::string_view dangling;
    // {
    //     std::string temp = "hello";
    //     dangling = temp; // dangling指向temp
    // } // temp销毁，dangling悬垂！

    return 0;
}
```

**输出：**
```
[LOG] C字符串字面量
[LOG] std::string
[LOG] string_view
```

## 三、注意事项与常见陷阱

- **`string_view`不拥有数据**：底层字符串销毁后视图悬垂。
- **不能从临时`string`创建`string_view`**：会悬垂。
- **不保证以`\0`结尾**：不能直接传给需要C字符串的函数。
- **`string_view`比`const string&`更通用**：接受C字符串和string。
- **修改底层数据会导致`string_view`失效**。
- **C++20新增`starts_with`/`ends_with`**。
