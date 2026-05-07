# 宽字符与Unicode

## 一、概念说明

`char`只能表示ASCII字符（0-127），无法处理中文、日文等多字节字符。C++提供了宽字符类型和Unicode支持来处理国际化文本。

## 二、具体用法

### 2.1 宽字符类型

```cpp
#include <iostream>
#include <string>
#include <locale>
using namespace std;

int main() {
    // char：1字节，ASCII字符
    char c = 'A';

    // wchar_t：平台相关大小（Windows 2字节，Linux 4字节）
    wchar_t wc = L'A';
    wstring ws = L"Hello 世界";

    // C++11固定大小Unicode类型
    // char16_t：2字节，UTF-16编码
    char16_t c16 = u'A';
    u16string s16 = u"Hello 世界";

    // char32_t：4字节，UTF-32编码
    char32_t c32 = U'A';
    u32string s32 = U"Hello 世界";

    // C++20：UTF-8编码
    // char8_t c8 = u8'A';    // C++20
    // u8string s8 = u8"Hello"; // C++20

    cout << "char大小:     " << sizeof(char) << " 字节" << endl;
    cout << "wchar_t大小:  " << sizeof(wchar_t) << " 字节" << endl;
    cout << "char16_t大小: " << sizeof(char16_t) << " 字节" << endl;
    cout << "char32_t大小: " << sizeof(char32_t) << " 字节" << endl;

    return 0;
}
```

输出（Linux）：
```
char大小:     1 字节
wchar_t大小:  4 字节
char16_t大小: 2 字节
char32_t大小: 4 字节
```

### 2.2 Unicode编码方案

```cpp
#include <iostream>
#include <string>
#include <codecvt>
#include <locale>
using namespace std;

int main() {
    // UTF-8：变长编码（1-4字节），最常用
    // ASCII字符1字节，中文字符3字节
    string utf8_str = "Hello你好";

    cout << "UTF-8字符串: " << utf8_str << endl;
    cout << "字节长度: " << utf8_str.length() << endl;  // 5+6=11字节
    cout << "字符长度（视觉）: 约7个字符" << endl;

    // UTF-16：Windows内部使用
    u16string utf16_str = u"Hello你好";

    // UTF-32：每个字符固定4字节，便于索引
    u32string utf32_str = U"Hello你好";
    cout << "UTF-32长度: " << utf32_str.length() << " 个字符" << endl;

    return 0;
}
```

输出：
```
UTF-8字符串: Hello你好
字节长度: 11
字符长度（视觉）: 约7个字符
UTF-32长度: 7 个字符
```

### 2.3 宽字符输出

```cpp
#include <iostream>
#include <string>
#include <locale>
using namespace std;

int main() {
    // 设置本地化以支持宽字符输出
    setlocale(LC_ALL, "");

    // Linux/macOS推荐使用UTF-8窄字符串
    string msg = "你好，世界！";
    cout << msg << endl;

    // 宽字符方式（Windows更常用）
    // wcout << L"你好" << endl;

    // C++20的std::format支持Unicode
    // print("你好{}", "世界");

    return 0;
}
```

输出：
```
你好，世界！
```

### 2.4 字符串字面量前缀

```cpp
#include <iostream>
using namespace std;

int main() {
    // 无前缀：char字符串
    auto s1 = "Hello";           // const char*

    // L前缀：wchar_t字符串
    auto s2 = L"Hello";          // const wchar_t*

    // u前缀：char16_t字符串（C++11）
    auto s3 = u"Hello";          // const char16_t*

    // U前缀：char32_t字符串（C++11）
    auto s4 = U"Hello";          // const char32_t*

    // u8前缀：UTF-8编码的char字符串（C++11）
    auto s5 = u8"Hello";         // const char*

    cout << "s1大小: " << sizeof(s1[0]) << " 字节/字符" << endl;
    cout << "s2大小: " << sizeof(s2[0]) << " 字节/字符" << endl;
    cout << "s3大小: " << sizeof(s3[0]) << " 字节/字符" << endl;
    cout << "s4大小: " << sizeof(s4[0]) << " 字节/字符" << endl;

    return 0;
}
```

输出：
```
s1大小: 1 字节/字符
s2大小: 4 字节/字符
s3大小: 2 字节/字符
s4大小: 4 字节/字符
```

## 三、注意事项与常见陷阱

1. **wchar_t大小不固定**：Windows上2字节（UTF-16），Linux上4字节（UTF-32），跨平台项目慎用
2. **UTF-8是推荐方案**：绝大多数场景使用`std::string` + UTF-8编码即可
3. **字符串长度**：`str.length()`返回字节数而非字符数，UTF-8中文每个字符占3字节
4. **Windows控制台编码**：Windows控制台默认GBK，需要设置代码页或使用宽字符API
5. **C++20改进**：C++20引入了`char8_t`和`std::format`，Unicode支持大幅改善
