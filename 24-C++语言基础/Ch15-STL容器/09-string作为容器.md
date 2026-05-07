# string作为容器

## 一、概念说明

`std::string`本质是一个存储`char`的特殊容器，具有连续存储、动态扩容等特性。它提供了完整的STL容器接口，可以与STL算法无缝配合。

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
    std::cout << "capacity: " << s.capacity() << std::endl;
    std::cout << "empty: " << s.empty() << std::endl;     // 0

    // 迭代器
    for (auto it = s.begin(); it != s.end(); ++it) {
        std::cout << *it;
    }
    std::cout << std::endl;

    // 随机访问
    s[0] = 'h';
    s.at(1) = 'E';

    // data()获取C风格字符串
    const char* cstr = s.data();
    std::cout << cstr << std::endl;
}
```

### 2.2 STL算法配合

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

    // 转换
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    std::cout << s << std::endl;  // DLROW OLLEH
}
```

### 2.3 string_view（C++17）

```cpp
#include <string_view>

void string_view_demo() {
    // string_view：非拥有视图，零拷贝
    std::string_view sv = "Hello, World!";
    std::cout << "sv: " << sv << std::endl;
    std::cout << "size: " << sv.size() << std::endl;

    // 子串（不分配内存）
    std::string_view sub = sv.substr(0, 5);
    std::cout << sub << std::endl;  // Hello

    // 从string创建
    std::string s = "test";
    std::string_view sv2(s);
}
```

## 三、注意事项与常见陷阱

- `std::string`是`std::basic_string<char>`的别名
- SSO（小字符串优化）：短字符串存在栈上，避免堆分配
- `c_str()`返回以null结尾的字符串，`data()`在C++11后也是
- `string_view`不拥有数据，原始数据必须在view生命周期内有效
- `reserve()`预分配空间，`shrink_to_fit()`释放多余空间
- string的迭代器失效规则与vector相同
