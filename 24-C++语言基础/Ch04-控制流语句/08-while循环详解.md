# while循环详解

## 一、概念说明

`while`循环在条件为true时重复执行循环体。它**先判断条件再执行**，循环体可能一次都不执行。适合**迭代次数不确定**的场景。

## 二、具体用法

### 2.1 基本while循环

```cpp
#include <iostream>
using namespace std;

int main() {
    // 基本while
    int i = 0;
    while (i < 5) {
        cout << i << " ";
        i++;
    }
    cout << endl;

    // 读取输入直到特定值
    int value;
    cout << "输入正数（输入0结束）:" << endl;
    // 模拟输入
    int inputs[] = {3, 7, 2, 0};
    int idx = 0;

    while (idx < 4) {
        value = inputs[idx++];
        if (value == 0) break;
        cout << "收到: " << value << endl;
    }

    // 查找条件
    int target = 42;
    int guess = 0;
    int attempts = 0;
    while (guess != target) {
        guess = (attempts + 1) * 10 + 2;  // 模拟猜测
        attempts++;
    }
    cout << "猜了" << attempts << "次找到" << target << endl;

    return 0;
}
```

输出：
```
0 1 2 3 4
输入正数（输入0结束）:
收到: 3
收到: 7
收到: 2
猜了4次找到42
```

### 2.2 与for循环的选择

```cpp
#include <iostream>
using namespace std;

int main() {
    // for适合：已知迭代次数
    for (int i = 0; i < 10; i++) {
        // ...
    }

    // while适合：条件驱动的循环
    bool running = true;
    int count = 0;
    while (running && count < 3) {
        cout << "运行中 " << count << endl;
        count++;
        if (count >= 3) running = false;
    }

    // while适合：文件读取、网络接收等
    // while (getline(file, line)) { ... }
    // while (connection.receive(data)) { ... }

    return 0;
}
```

输出：
```
运行中 0
运行中 1
运行中 2
```

## 三、注意事项与常见陷阱

1. **死循环**：条件永远为true导致无限循环，确保循环内有使条件变false的操作
2. **循环变量初始化**：`while`不在循环头声明变量，忘记初始化可能导致问题
3. **先判断后执行**：条件初始为false时循环体一次都不执行
4. **与do-while的区别**：do-while至少执行一次
5. **空循环体**：`while (condition);`末尾分号可能导致空循环体
