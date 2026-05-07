# 字符类型 char

## 一、概念说明

`char`类型用于存储单个字符，实际存储的是字符对应的**ASCII码值**（一个整数）。C++中`char`有三种变体：`char`、`signed char`和`unsigned char`。

## 二、具体用法

### 2.1 char的基本使用

```cpp
#include <iostream>
using namespace std;

int main() {
    // 字符字面量
    char ch1 = 'A';
    char ch2 = 65;          // ASCII码65就是'A'
    char ch3 = '\x41';      // 十六进制表示
    char ch4 = '\101';      // 八进制表示

    cout << "ch1: " << ch1 << endl;
    cout << "ch2: " << ch2 << endl;
    cout << "ch3: " << ch3 << endl;
    cout << "ch4: " << ch4 << endl;

    // char的整数值
    cout << "'A'的ASCII码: " << (int)ch1 << endl;
    cout << "'a'的ASCII码: " << (int)'a' << endl;
    cout << "'0'的ASCII码: " << (int)'0' << endl;

    // 大小写转换
    char upper = 'A';
    char lower = upper + 32;  // 大写转小写
    cout << "大写: " << upper << " -> 小写: " << lower << endl;

    return 0;
}
```

输出：
```
ch1: A
ch2: A
ch3: A
ch4: A
'A'的ASCII码: 65
'a'的ASCII码: 97
'0'的ASCII码: 48
大写: A -> 小写: a
```

### 2.2 转义字符

```cpp
#include <iostream>
using namespace std;

int main() {
    // 常用转义字符
    cout << "换行: Hello\nWorld" << endl;
    cout << "制表: Name\tAge\tScore" << endl;
    cout << "反斜杠: C:\\Users\\file.txt" << endl;
    cout << "单引号: It\'s OK" << endl;
    cout << "双引号: She said \"Hi\"" << endl;
    cout << "空字符: " << 'X' << '\0' << 'Y' << endl;  // Y不会显示
    cout << "响铃: " << '\a' << endl;

    // 转义字符汇总
    // \n 换行     \t 水平制表   \\ 反斜杠
    // \' 单引号   \" 双引号     \0 空字符
    // \r 回车     \a 响铃       \b 退格
    // \xhh 十六进制  \ooo 八进制

    return 0;
}
```

输出：
```
换行: Hello
World
制表: Name    Age     Score
反斜杠: C:\Users\file.txt
单引号: It's OK
双引号: She said "Hi"
空字符: X
响铃:
```

### 2.3 signed char vs unsigned char vs char

```cpp
#include <iostream>
#include <climits>
using namespace std;

int main() {
    // signed char: -128 ~ 127
    signed char sc = -50;
    // unsigned char: 0 ~ 255
    unsigned char uc = 200;
    // char: 符号性取决于编译器实现
    char ch = 128;  // 可能溢出（如果有符号）

    cout << "signed char范围: " << SCHAR_MIN << " ~ " << SCHAR_MAX << endl;
    cout << "unsigned char范围: 0 ~ " << UCHAR_MAX << endl;
    cout << "char大小: " << sizeof(char) << " 字节（始终为1）" << endl;

    // 用char做整数运算可能出问题
    char a = 127;
    a++;  // 如果是有符号char，这是溢出（未定义行为）
    cout << "127 + 1 = " << (int)a << endl;

    return 0;
}
```

输出（有符号char）：
```
signed char范围: -128 ~ 127
unsigned char范围: 0 ~ 255
char大小: 1 字节（始终为1）
127 + 1 = -128
```

### 2.4 字符判断函数

```cpp
#include <iostream>
#include <cctype>
using namespace std;

int main() {
    char tests[] = {'A', 'z', '5', ' ', '\n', '@'};

    for (char c : tests) {
        cout << "'" << c << "' (ASCII " << (int)c << "):" << endl;
        cout << "  isalpha: " << (isalpha(c) ? "是字母" : "非字母");
        cout << ", isdigit: " << (isdigit(c) ? "是数字" : "非数字");
        cout << ", isalnum: " << (isalnum(c) ? "是字母或数字" : "否");
        cout << ", isspace: " << (isspace(c) ? "是空白" : "非空白");
        cout << endl;
    }

    // 大小写转换
    cout << "\ntolower('A') = " << (char)tolower('A') << endl;
    cout << "toupper('z') = " << (char)toupper('z') << endl;

    return 0;
}
```

输出：
```
'A' (ASCII 65):
  isalpha: 是字母, isdigit: 非数字, isalnum: 是字母或数字, isspace: 非空白
'z' (ASCII 122):
  isalpha: 是字母, isdigit: 非数字, isalnum: 是字母或数字, isspace: 非空白
'5' (ASCII 53):
  isalpha: 非字母, isdigit: 是数字, isalnum: 是字母或数字, isspace: 非空白
```

## 三、注意事项与常见陷阱

1. **char是有符号还是无符号是实现定义的**：不要依赖char的符号性，需要时显式指定
2. **单引号vs双引号**：`'A'`是char，`"A"`是包含null终止符的字符数组
3. **char参与算术运算**：char会被提升为int参与运算，结果可能溢出char范围
4. **中文不是单个char**：中文字符通常需要多个字节，使用`std::wstring`或UTF-8编码
5. **用<cctype>函数判断字符属性**：不要用`c >= 'A' && c <= 'Z'`，用`isupper(c)`
