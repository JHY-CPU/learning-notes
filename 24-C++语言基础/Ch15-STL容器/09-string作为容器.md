# string作为容器

## 一、概念说明

`std::string`本质是`std::basic_string<char>`的特化（C++标准 §24.3.2），是一个存储`char`的特殊容器。它具有连续存储、动态扩容等特性，提供了完整的STL容器接口，可以与STL算法无缝配合。string还具有SSO（小字符串优化）等特殊优化。

### 1.1 string与vector<char>的差异

| 特性 | string | vector<char> |
|------|--------|-------------|
| SSO优化 | 有（短字符串栈上存储） | 无 |
| 字符串操作 | 丰富（find, substr等） | 无 |
| C接口 | c_str()直接可用 | 需要手动加'\0' |
| 语义 | 字符串 | 字符集合 |

## 二、具体用法

### 2.1 容器接口

```cpp
#include <string>
#include <iostream>
#include <algorithm>

int main() {
    std::string s = "Hello, World!";

    // 容器接口
    std::cout << "size: " << s.size() << std::endl;       // 13
    std::cout << "length: " << s.length() << std::endl;   // 同size
    std::cout << "capacity: " << s.capacity() << std::endl;
    std::cout << "empty: " << s.empty() << std::endl;     // 0

    // 随机访问
    s[0] = 'h';
    s.at(1) = 'E';  // 越界检查

    // 首尾
    std::cout << "front: " << s.front() << std::endl;
    std::cout << "back: " << s.back() << std::endl;

    // 迭代器
    for (auto it = s.begin(); it != s.end(); ++it)
        std::cout << *it;
    std::cout << std::endl;

    // data()和c_str()（C++11后两者等价，都以'\0'结尾）
    const char* cstr = s.c_str();
    const char* data = s.data();

    // reserve预分配
    s.reserve(100);
    s.shrink_to_fit();
}
```

### 2.2 字符串操作

```cpp
void string_ops() {
    std::string s = "Hello, World!";

    // 查找
    auto pos = s.find("World");         // 7
    pos = s.find('o');                  // 4
    pos = s.find("xyz");                // npos（未找到）

    // 子串
    std::string sub = s.substr(7, 5);   // "World"
    sub = s.substr(7);                  // "World!"

    // 替换
    s.replace(7, 5, "C++");            // "Hello, C++!"

    // 插入
    s.insert(5, " Beautiful");         // "Hello Beautiful, C++!"

    // 删除
    s.erase(5, 10);                    // "Hello, C++!"

    // 追加
    s += " is great";
    s.append("!");
    s.push_back('!');

    // 比较
    if (s == "Hello") { }
    if (s.compare("Hello") == 0) { }

    // 查找所有出现位置
    std::string text = "abcabcabc";
    size_t found = 0;
    while ((found = text.find("abc", found)) != std::string::npos) {
        std::cout << "找到: " << found << std::endl;
        found += 3;
    }
}
```

### 2.3 STL算法配合

```cpp
void with_algorithms() {
    std::string s = "hello world";

    // 排序
    std::sort(s.begin(), s.end());

    // 查找
    auto it = std::find(s.begin(), s.end(), 'w');

    // 反转
    std::reverse(s.begin(), s.end());

    // 统计
    int count = std::count(s.begin(), s.end(), 'l');

    // 转换大写
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);

    // 删除所有空格（C++20 erase/erase_if）
    // std::erase(s, ' ');
    // 或传统方式
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
}
```

### 2.4 string_view（C++17）

```cpp
#include <string_view>

void string_view_demo() {
    // string_view：非拥有视图，零拷贝
    std::string_view sv = "Hello, World!";  // 不拷贝字符串
    std::cout << "sv: " << sv << std::endl;
    std::cout << "size: " << sv.size() << std::endl;

    // 子串（不分配内存）
    std::string_view sub = sv.substr(0, 5);  // "Hello"

    // 从string创建（安全）
    std::string s = "test";
    std::string_view sv2(s);

    // 从C字符串创建
    std::string_view sv3("literal");

    // 注意：string_view不拥有数据！
    // 危险：返回local string的view
    // std::string_view bad() {
    //     std::string local = "oops";
    //     return local;  // 悬空引用！
    // }

    // 函数参数推荐用string_view
    auto process = [](std::string_view sv) {
        std::cout << sv << std::endl;
    };
    process("literal");  // OK
    process(s);          // OK
}
```

## 三、SSO（小字符串优化）

```cpp
void sso_demo() {
    std::string short_str = "short";       // SSO：存在栈上
    std::string long_str(100, 'x');        // 超过SSO阈值：存在堆上

    std::cout << "short capacity: " << short_str.capacity() << std::endl;
    // 通常15或22（取决于实现）

    std::cout << "long capacity: " << long_str.capacity() << std::endl;
    // >= 100

    // SSO的好处：短字符串无堆分配，更快
    // 大多数字符串是短的（文件名、标识符等）
}
```

## 四、注意事项与常见陷阱

1. **`std::string`是`std::basic_string<char>`的别名**
2. **SSO（小字符串优化）**：短字符串存在栈上，避免堆分配，性能优于`vector<char>`
3. **`c_str()`和`data()`在C++11后都返回以'\0'结尾的字符串**
4. **`string_view`不拥有数据**：原始数据必须在view生命周期内有效
5. **`reserve()`预分配空间**，`shrink_to_fit()`释放多余空间
6. **迭代器失效规则与vector相同**：扩容时全部失效
7. **`std::string`可能不是UTF-8安全的**：处理多字节字符用`std::u8string`（C++20）或第三方库
