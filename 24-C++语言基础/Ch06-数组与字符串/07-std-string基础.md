# std::string基础

## 一、概念说明

`std::string`是C++标准库提供的字符串类，定义在`<string>`头文件中。它自动管理内存，支持动态增长，提供了丰富的操作方法。相比C风格字符串，`std::string`更安全、更方便。

## 二、具体用法

### 2.1 构造方式

```cpp
#include <string>

std::string s1;                          // 空字符串
std::string s2("Hello");                 // 从字面量构造
std::string s3(5, 'A');                  // "AAAAA"（5个A）
std::string s4 = "World";               // 拷贝初始化
std::string s5(s2);                      // 拷贝构造
std::string s6(s2, 1, 3);               // 子串：从位置1取3个 → "ell"

std::cout << s2 << std::endl;  // 输出: Hello
std::cout << s3 << std::endl;  // 输出: AAAAA
std::cout << s6 << std::endl;  // 输出: ell
```

### 2.2 赋值与比较

```cpp
std::string s;
s = "New Value";        // 赋值
s += " appended";       // 追加赋值
s.append(" more");      // 追加

std::cout << s << std::endl;
// 输出: New Value appended more

// 比较
std::string a = "apple", b = "banana";
std::cout << (a < b) << std::endl;   // 输出: 1 (true)
std::cout << a.compare(b) << std::endl;  // 输出: 负数 (a < b)
```

### 2.3 连接

```cpp
std::string first = "Hello";
std::string second = " World";

// + 运算符
std::string combined = first + second;
std::cout << combined << std::endl;  // 输出: Hello World

// + 字面量
std::string with_literal = first + " there!";
std::cout << with_literal << std::endl;  // 输出: Hello there!
```

### 2.4 常用属性

```cpp
std::string s = "Hello, C++!";
std::cout << "长度: " << s.size() << std::endl;      // 输出: 11
std::cout << "长度: " << s.length() << std::endl;    // 输出: 11
std::cout << "容量: " << s.capacity() << std::endl;  // 输出: >= 11
std::cout << "是否空: " << s.empty() << std::endl;   // 输出: 0

s.clear();
std::cout << "清空后empty: " << s.empty() << std::endl;  // 输出: 1
```

### 2.5 元素访问

```cpp
std::string s = "Hello";
std::cout << s[0] << std::endl;     // 输出: H
std::cout << s.at(4) << std::endl;  // 输出: o
std::cout << s.front() << std::endl;  // 输出: H
std::cout << s.back() << std::endl;   // 输出: o
```

## 三、注意事项与常见陷阱

- `operator[]`不检查越界，`at()`越界抛出`std::out_of_range`
- `std::string`可以包含`\0`字符（不同于C字符串）
- `c_str()`返回的指针在string修改后可能失效
- `size()`和`length()`功能完全相同
- 字符串拼接注意临时对象的效率问题
