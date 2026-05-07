# switch进阶技巧

## 一、概念说明

本节介绍switch语句的高级用法：枚举switch、C++17初始化语句、查表替代等优化技巧。

## 二、具体用法

### 2.1 枚举switch

```cpp
#include <iostream>
using namespace std;

enum class Command { Open, Close, Read, Write, Invalid };

Command parseCommand(const string& cmd) {
    if (cmd == "open") return Command::Open;
    if (cmd == "close") return Command::Close;
    if (cmd == "read") return Command::Read;
    if (cmd == "write") return Command::Write;
    return Command::Invalid;
}

int main() {
    Command cmd = parseCommand("read");

    switch (cmd) {
        case Command::Open:
            cout << "打开文件" << endl;
            break;
        case Command::Close:
            cout << "关闭文件" << endl;
            break;
        case Command::Read:
            cout << "读取数据" << endl;
            break;
        case Command::Write:
            cout << "写入数据" << endl;
            break;
        case Command::Invalid:
            cout << "未知命令" << endl;
            break;
    }

    return 0;
}
```

输出：
```
读取数据
```

### 2.2 C++17 switch初始化语句

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    // C++17：switch中包含初始化
    switch (string input = "hello"; input[0]) {
        case 'h':
            cout << "以h开头" << endl;
            break;
        case 'w':
            cout << "以w开头" << endl;
            break;
        default:
            cout << "其他" << endl;
    }
    // input在这里不可见

    // 实用场景：解析状态
    int nextState = 3;
    switch (int state = nextState; state) {
        case 0: cout << "初始状态" << endl; break;
        case 1: cout << "运行中" << endl; break;
        case 2: cout << "暂停" << endl; break;
        case 3: cout << "完成" << endl; break;
    }

    return 0;
}
```

输出：
```
以h开头
完成
```

### 2.3 switch vs 查表

```cpp
#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;

int main() {
    // switch方式
    auto switchMethod = [](int code) -> string {
        switch (code) {
            case 200: return "OK";
            case 404: return "Not Found";
            case 500: return "Server Error";
            default: return "Unknown";
        }
    };

    // 查表方式（更灵活）
    static const unordered_map<int, string> statusCodes = {
        {200, "OK"},
        {404, "Not Found"},
        {500, "Server Error"}
    };

    auto lookupMethod = [&](int code) -> string {
        auto it = statusCodes.find(code);
        return (it != statusCodes.end()) ? it->second : "Unknown";
    };

    cout << switchMethod(200) << endl;
    cout << lookupMethod(404) << endl;

    return 0;
}
```

输出：
```
OK
Not Found
```

## 三、注意事项与常见陷阱

1. **switch不适合大量case**：case超过10个考虑用查表或策略模式
2. **枚举类配合switch**：编译器可检查是否覆盖了所有枚举值
3. **初始化语句限制作用域**：避免变量名称泄漏到switch外
4. **查表更灵活**：case值可以是运行时确定的，switch必须编译期常量
5. **性能**：少量case时switch通常比查表快，大量case时查表更优
