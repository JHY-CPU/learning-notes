# std::string查找操作

## 一、概念说明

`std::string`提供了一系列查找方法：正向查找`find`、反向查找`rfind`、查找任意字符`find_first_of`、查找不在集合中的字符`find_first_not_of`等。所有查找失败时返回`std::string::npos`（通常是`-1`的最大无符号值）。

## 二、具体用法

### 2.1 find 和 rfind

```cpp
std::string s = "Hello World Hello C++";

// 正向查找子串
size_t pos = s.find("Hello");
std::cout << "第一次出现: " << pos << std::endl;
// 输出: 第一次出现: 0

// 从指定位置开始查找
pos = s.find("Hello", 1);
std::cout << "从位置1起: " << pos << std::endl;
// 输出: 从位置1起: 12

// 反向查找
pos = s.rfind("Hello");
std::cout << "最后出现: " << pos << std::endl;
// 输出: 最后出现: 12
```

### 2.2 find_first_of / find_last_of

```cpp
std::string s = "Hello World";

// 查找任意指定字符首次出现
pos = s.find_first_of("aeiou");
std::cout << "第一个元音位置: " << pos << std::endl;
// 输出: 第一个元音位置: 1 ('e')

// 查找数字
std::string data = "abc123def";
pos = data.find_first_of("0123456789");
std::cout << "第一个数字位置: " << pos << std::endl;
// 输出: 第一个数字位置: 3

// 查找最后出现的指定字符
pos = s.find_last_of("lo");
std::cout << "最后的'l'或'o': " << pos << std::endl;
// 输出: 最后的'l'或'o': 9
```

### 2.3 find_first_not_of / find_last_not_of

```cpp
std::string s = "   Hello   ";

// 查找第一个非空格字符
size_t start = s.find_first_not_of(' ');
std::cout << "首非空格: " << start << std::endl;
// 输出: 首非空格: 3

// 查找最后一个非空格字符
size_t end = s.find_last_not_of(' ');
std::cout << "末非空格: " << end << std::endl;
// 输出: 末非空格: 7

// 去除首尾空格
std::string trimmed = s.substr(start, end - start + 1);
std::cout << "'" << trimmed << "'" << std::endl;
// 输出: 'Hello'
```

### 2.4 检查查找结果

```cpp
std::string s = "Hello";
if (s.find("xyz") == std::string::npos) {
    std::cout << "未找到\n";
}
// 输出: 未找到
```

## 三、注意事项与常见陷阱

- 查找失败返回`std::string::npos`，不要用`== -1`比较（有符号/无符号问题）
- `find`的第二个参数是起始搜索位置
- `find_first_of`查找集合中**任意一个**字符，不是子串
- `rfind`从后往前找，但返回的仍是正向位置
- 每次查找都是O(n)复杂度，频繁查找考虑其他数据结构
