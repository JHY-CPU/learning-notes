# std::string修改操作

## 一、概念说明

`std::string`提供了丰富的修改操作：插入（`insert`）、删除（`erase`）、替换（`replace`）、追加（`append`）。这些操作会动态调整字符串的大小。

## 二、具体用法

### 2.1 insert（插入）

```cpp
std::string s = "Hello World";

s.insert(5, ",");        // 在位置5插入逗号
std::cout << s << std::endl;  // 输出: Hello, World

s.insert(6, " Beautiful");  // 在位置6插入子串
std::cout << s << std::endl;
// 输出: Hello, Beautiful World

// 插入重复字符
s.insert(0, 3, '*');
std::cout << s << std::endl;
// 输出: ***Hello, Beautiful World
```

### 2.2 erase（删除）

```cpp
std::string s = "Hello Beautiful World";

s.erase(5, 10);  // 从位置5删除10个字符
std::cout << s << std::endl;  // 输出: Hello World

// 删除到末尾
s.erase(5);
std::cout << s << std::endl;  // 输出: Hello

// 删除指定位置字符
s.erase(s.begin() + 1);
std::cout << s << std::endl;  // 输出: Hllo
```

### 2.3 replace（替换）

```cpp
std::string s = "Hello World";

s.replace(6, 5, "C++");  // 位置6起的5个字符替换为"C++"
std::cout << s << std::endl;  // 输出: Hello C++

// 替换为重复字符
s.replace(0, 5, 3, 'X');
std::cout << s << std::endl;  // 输出: XXX C++
```

### 2.4 append（追加）

```cpp
std::string s = "Hello";

s.append(" World");          // 追加字符串
s.append(3, '!');             // 追加3个'!'
s.append(" C++", 1, 3);     // 追加子串

std::cout << s << std::endl;
// 输出: Hello World!!! C++

// += 是 append 的语法糖
s += " rocks";
std::cout << s << std::endl;
// 输出: Hello World!!! C++ rocks
```

### 2.5 substr（子串）

```cpp
std::string s = "Hello World";
std::string sub = s.substr(6, 5);  // 位置6起取5个字符
std::cout << sub << std::endl;  // 输出: World

// 从位置6到末尾
std::string rest = s.substr(6);
std::cout << rest << std::endl;  // 输出: World
```

## 三、注意事项与常见陷阱

- `insert`和`erase`可能导致大量字符移动，频繁操作效率低
- `substr`创建新字符串，有拷贝开销
- 位置参数超出范围会抛出`std::out_of_range`
- `replace`的第二个参数是**替换长度**，不是结束位置
- 批量修改建议先构建再赋值，避免多次修改
