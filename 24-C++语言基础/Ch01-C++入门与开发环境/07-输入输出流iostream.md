# 输入输出流 iostream

## 一、概念说明

C++使用**流（stream）**进行输入输出操作。流是一种抽象，代表数据从源到目标的流动。`<iostream>`库提供了四个标准流对象：`cin`、`cout`、`cerr`和`clog`。

## 二、具体用法

### 2.1 标准输出 cout

```cpp
#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    int age = 25;
    double pi = 3.14159265358979;

    // 基本输出
    cout << "年龄: " << age << endl;

    // 链式输出
    cout << "姓名: " << "张三" << ", 年龄: " << age << endl;

    // 格式化输出
    cout << fixed << setprecision(4);  // 固定4位小数
    cout << "PI = " << pi << endl;

    // 设置宽度和填充
    cout << setfill('0') << setw(8) << 42 << endl;
    return 0;
}
```

输出：
```
年龄: 25
姓名: 张三, 年龄: 25
PI = 3.1416
00000042
```

### 2.2 标准输入 cin

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    int number;
    string name;

    // 读取整数
    cout << "请输入一个整数: ";
    cin >> number;
    cout << "你输入的是: " << number << endl;

    // 读取字符串（遇空格停止）
    cout << "请输入一个单词: ";
    cin >> name;
    cout << "你输入的是: " << name << endl;

    // 读取一行（含空格）
    cin.ignore();  // 清除缓冲区中的换行符
    string line;
    cout << "请输入一行文字: ";
    getline(cin, line);
    cout << "你输入的是: " << line << endl;

    return 0;
}
```

运行示例：
```
请输入一个整数: 42
你输入的是: 42
请输入一个单词: Hello
你输入的是: Hello
请输入一行文字: Hello World
你输入的是: Hello World
```

### 2.3 cerr 和 clog

```cpp
#include <iostream>
using namespace std;

int main() {
    // cerr: 标准错误输出，无缓冲，立即输出
    cerr << "错误: 文件未找到!" << endl;

    // clog: 标准日志输出，有缓冲
    clog << "信息: 程序开始执行" << endl;
    clog << "信息: 正在处理数据..." << endl;

    // cout: 标准输出，有缓冲
    cout << "正常输出内容" << endl;
    return 0;
}
```

输出：
```
正常输出内容
错误: 文件未找到!
信息: 程序开始执行
信息: 正在处理数据...
```

### 2.4 endl vs '\n'

```cpp
#include <iostream>
using namespace std;

int main() {
    // endl: 输出换行符并刷新缓冲区
    cout << "使用endl" << endl;

    // '\n': 仅输出换行符，不刷新缓冲区
    cout << "使用\\n" << '\n';

    // 性能对比：大量输出时'\n'更快
    // 因为endl每次都要刷新缓冲区
    for (int i = 0; i < 5; i++) {
        cout << "行" << i << '\n';  // 推荐
    }
    return 0;
}
```

输出：
```
使用endl
使用\n
行0
行1
行2
行3
行4
```

### 2.5 格式化输出

```cpp
#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    double value = 12345.6789;

    // 设置小数位数
    cout << fixed << setprecision(2);
    cout << "金额: " << value << endl;

    // 科学计数法
    cout << scientific << setprecision(3);
    cout << "科学: " << value << endl;

    // 十六进制输出
    int num = 255;
    cout << "十进制: " << dec << num << endl;
    cout << "十六进制: " << hex << num << endl;
    cout << "八进制: " << oct << num << endl;

    // 左对齐和右对齐
    cout << left << setw(20) << "左对齐" << "|" << endl;
    cout << right << setw(20) << "右对齐" << "|" << endl;
    return 0;
}
```

输出：
```
金额: 12345.68
科学: 1.235e+04
十进制: 255
十六进制: ff
八进制: 377
左对齐              |
              右对齐|
```

## 三、注意事项与常见陷阱

1. **cin >> 遇空格停止**：`cin >> str`只会读取到第一个空白字符，读取含空格的字符串应用`getline(cin, str)`
2. **输入失败时cin的状态**：当输入类型不匹配时，`cin`进入失败状态，后续所有输入都会失败，需要调用`cin.clear()`和`cin.ignore()`
3. **缓冲区问题**：混合使用`cin >>`和`getline`时，`getline`会读取到残留在缓冲区的换行符，需要先`cin.ignore()`
4. **不要用endl大量输出**：频繁刷新缓冲区会严重影响性能，用`'\n'`代替
5. **输出顺序**：`cerr`和`cout`的输出顺序可能与预期不同，因为它们的缓冲策略不同
