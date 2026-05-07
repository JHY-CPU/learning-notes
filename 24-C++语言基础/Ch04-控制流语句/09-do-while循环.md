# do-while循环

## 一、概念说明

`do-while`循环**先执行循环体再判断条件**，保证循环体至少执行一次。适合需要先执行再判断的场景，如菜单系统、输入验证。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
using namespace std;

int main() {
    // 基本do-while
    int i = 0;
    do {
        cout << i << " ";
        i++;
    } while (i < 5);
    cout << endl;

    // 即使条件初始为false，也执行一次
    int x = 10;
    do {
        cout << "即使x=10也会执行一次, x=" << x << endl;
    } while (x < 5);

    // 菜单系统
    int choice;
    int menuInput = 3;  // 模拟用户输入
    int inputIdx = 0;
    int inputs[] = {1, 2, 3};  // 模拟输入序列

    do {
        choice = inputs[inputIdx++];
        cout << "选择了: " << choice << endl;
    } while (choice != 3);

    return 0;
}
```

输出：
```
0 1 2 3 4
即使x=10也会执行一次, x=10
选择了: 1
选择了: 2
选择了: 3
```

### 2.2 输入验证

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    // 输入验证：先读取，再检查
    string input = "abc";  // 模拟无效输入
    int attempts = 0;
    string validInputs[] = {"abc", "abc", "valid"};
    int idx = 0;

    do {
        input = validInputs[idx++];
        attempts++;
        cout << "第" << attempts << "次输入: " << input << endl;
    } while (input != "valid" && attempts < 5);

    if (input == "valid") {
        cout << "输入有效" << endl;
    } else {
        cout << "达到最大尝试次数" << endl;
    }

    return 0;
}
```

输出：
```
第1次输入: abc
第2次输入: abc
第3次输入: valid
输入有效
```

## 三、注意事项与常见陷阱

1. **末尾分号**：`while (condition);`必须有分号
2. **至少执行一次**：如果不需要这个特性，用while更合适
3. **适用于菜单和输入验证**：先做再判断的模式
4. **避免滥用**：大多数循环用for或while即可
5. **死循环风险**：确保循环内有退出条件
