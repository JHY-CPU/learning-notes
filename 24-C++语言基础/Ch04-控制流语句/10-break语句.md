# break语句

## 一、概念说明

`break`立即**跳出当前循环或switch语句**，执行后续代码。它只跳出最近的一层循环。

## 二、具体用法

### 2.1 在循环中使用

```cpp
#include <iostream>
using namespace std;

int main() {
    // 找到第一个能被7整除的数
    for (int i = 1; i <= 100; i++) {
        if (i % 7 == 0) {
            cout << "找到: " << i << endl;
            break;  // 找到后立即退出
        }
    }

    // while中的break
    int sum = 0;
    int i = 0;
    while (true) {  // 无限循环
        sum += i;
        if (sum > 100) {
            cout << "累加超过100: sum=" << sum << ", i=" << i << endl;
            break;
        }
        i++;
    }

    // do-while中的break
    int x = 0;
    do {
        x++;
        if (x > 3) break;
        cout << x << " ";
    } while (true);
    cout << endl;

    return 0;
}
```

输出：
```
找到: 7
累加超过100: sum=105, i=14
1 2 3
```

### 2.2 在嵌套循环中

```cpp
#include <iostream>
using namespace std;

int main() {
    // break只跳出一层循环
    for (int i = 0; i < 3; i++) {
        cout << "外层 i=" << i << ": ";
        for (int j = 0; j < 5; j++) {
            if (j == 3) break;  // 只跳出内层
            cout << j << " ";
        }
        cout << endl;
    }

    // 使用标志变量跳出多层
    bool found = false;
    for (int i = 0; i < 3 && !found; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == 1 && j == 2) {
                cout << "找到 (" << i << "," << j << ")" << endl;
                found = true;
                break;
            }
        }
    }

    return 0;
}
```

输出：
```
外层 i=0: 0 1 2
外层 i=1: 0 1 2
外层 i=2: 0 1 2
找到 (1,2)
```

## 三、注意事项与常见陷阱

1. **break只跳出一层**：嵌套循环中需要标志变量或goto跳出多层
2. **switch中的break**：防止case贯穿，不要忘记
3. **break后代码不可达**：break之后的语句不会执行
4. **无限循环的退出**：`while(true)`通常用break退出
5. **提前退出优化**：找到结果后立即break，避免无意义的迭代
