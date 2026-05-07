# switch语句详解

## 一、概念说明

`switch`语句根据表达式的值选择多个分支中的一个执行。它比多个`if-else if`更清晰，但仅适用于离散值（整型、枚举）的多路分支。

## 二、具体用法

### 2.1 基本switch

```cpp
#include <iostream>
using namespace std;

int main() {
    int day = 3;

    switch (day) {
        case 1:
            cout << "星期一" << endl;
            break;
        case 2:
            cout << "星期二" << endl;
            break;
        case 3:
            cout << "星期三" << endl;
            break;
        case 4:
            cout << "星期四" << endl;
            break;
        case 5:
            cout << "星期五" << endl;
            break;
        case 6:
        case 7:
            cout << "周末" << endl;
            break;
        default:
            cout << "无效日期" << endl;
            break;
    }

    return 0;
}
```

输出：
```
星期三
```

### 2.2 fall-through（贯穿）

```cpp
#include <iostream>
using namespace std;

int main() {
    // 故意的fall-through：统计字母频率
    char ch = 'a';
    int count = 0;

    switch (ch) {
        case 'a': count++;  // 注意：这里没有break，会贯穿！
        case 'e': count++;
        case 'i': count++;
        case 'o': count++;
        case 'u': count++;
            cout << "经过了 " << count << " 个case" << endl;
            break;
    }
    // 输出5（所有case都被执行了）

    // C++17：使用[[fallthrough]]属性标记有意的贯穿
    switch (ch) {
        case 'a':
            count = 1;
            [[fallthrough]];  // 告诉编译器：这是有意的贯穿
        case 'b':
            count += 1;
            break;
    }
    cout << "最终count: " << count << endl;

    return 0;
}
```

输出：
```
经过了 5 个case
最终count: 2
```

### 2.3 switch的限制

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    // switch只能用于整型和枚举
    // switch (string s) { }  // 编译错误：不能用于string
    // switch (3.14) { }      // 编译错误：不能用于浮点

    // case值必须是编译期常量
    int x = 3;
    // switch (x) {
    //     case y:  // 如果y不是常量，编译错误
    // }

    // 可以用的类型
    switch (x) {
        case 1: break;
        case 2: break;
        case 3: break;
    }

    // 枚举
    enum class Color { Red, Green, Blue };
    Color c = Color::Green;
    switch (c) {
        case Color::Red:   cout << "红" << endl; break;
        case Color::Green: cout << "绿" << endl; break;
        case Color::Blue:  cout << "蓝" << endl; break;
    }

    return 0;
}
```

输出：
```
绿
```

## 三、注意事项与常见陷阱

1. **忘记break**：最常见错误，导致意外贯穿
2. **default不能省略**：即使认为已覆盖所有情况，也应包含default
3. **case值必须是常量**：不能用变量作为case标签
4. **不能用于浮点和字符串**：这些类型用if-else处理
5. **[[fallthrough]]属性**：C++17中用它标记有意的贯穿，避免编译器警告
