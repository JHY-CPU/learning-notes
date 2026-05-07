# init-statement条件语句

## 一、概念说明

C++17允许在`if`和`switch`语句中包含一个**初始化语句**，在条件判断之前执行。这限制了变量的作用域，使代码更紧凑。

## 二、具体用法

### 2.1 if中的初始化语句

```cpp
#include <iostream>
#include <map>
#include <string>
using namespace std;

int main() {
    map<string, int> scores = {{"Alice", 95}, {"Bob", 87}};

    // 传统方式
    auto it = scores.find("Alice");
    if (it != scores.end()) {
        cout << it->first << ": " << it->second << endl;
    }
    // it仍然在作用域内（可能被误用）

    // C++17方式：变量只在if-else中可见
    if (auto it2 = scores.find("Alice"); it2 != scores.end()) {
        cout << "找到: " << it2->first << endl;
    } else {
        cout << "未找到，it2在这里也可用" << endl;
    }
    // it2在这里不可见

    // 实用场景：锁的作用域
    // if (lock_guard<mutex> lock(mtx); dataReady) {
    //     processData();
    // }

    return 0;
}
```

输出：
```
Alice: 95
找到: Alice
```

### 2.2 switch中的初始化语句

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    // switch中的初始化
    string cmd = "start";

    switch (auto first = cmd[0]; first) {
        case 's':
            cout << "start/stop命令" << endl;
            break;
        case 'q':
            cout << "quit命令" << endl;
            break;
        default:
            cout << "未知命令，首字符: " << first << endl;
    }
    // first在这里不可见

    return 0;
}
```

输出：
```
start/stop命令
```

## 三、注意事项与常见陷阱

1. **变量作用域限制**：初始化的变量只在if/switch和对应的else/case中可见
2. **可以用逗号**：`if (int x = 1, y = 2; x + y > 0)`是合法的
3. **类型推导**：`if (auto x = func(); x > 0)`自动推导类型
4. **C++17特性**：需要`-std=c++17`
5. **不是必须的**：传统写法同样正确，这只是语法糖
