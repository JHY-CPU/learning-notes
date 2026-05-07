# continue语句

## 一、概念说明

`continue`跳过当前迭代的**剩余部分**，直接进入下一次迭代。它只影响最近的一层循环。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
using namespace std;

int main() {
    // 跳过偶数，只打印奇数
    for (int i = 1; i <= 10; i++) {
        if (i % 2 == 0) continue;
        cout << i << " ";
    }
    cout << endl;

    // 跳过特定值
    int nums[] = {1, -2, 3, -4, 5, 0, 7};
    int sum = 0;
    for (int n : nums) {
        if (n <= 0) continue;  // 跳过非正数
        sum += n;
    }
    cout << "正数之和: " << sum << endl;

    // 在while中使用
    int i = 0;
    while (i < 5) {
        i++;
        if (i == 3) continue;
        cout << i << " ";
    }
    cout << endl;

    return 0;
}
```

输出：
```
1 3 5 7 9
正数之和: 16
1 2 4 5
```

### 2.2 嵌套循环中的continue

```cpp
#include <iostream>
using namespace std;

int main() {
    // continue只影响内层循环
    for (int i = 0; i < 3; i++) {
        cout << "行" << i << ": ";
        for (int j = 0; j < 5; j++) {
            if (j == 2) continue;  // 跳过j==2
            cout << j << " ";
        }
        cout << endl;
    }

    return 0;
}
```

输出：
```
行0: 0 1 3 4
行1: 0 1 3 4
行2: 0 1 3 4
```

## 三、注意事项与常见陷阱

1. **continue后代码不执行**：当前迭代中continue之后的语句被跳过
2. **while循环注意迭代更新**：`while`中continue可能跳过`i++`导致死循环
3. **只影响一层**：嵌套循环中continue不影响外层
4. **可读性**：避免在循环开头连续多个continue，考虑用if重写
5. **与break区别**：continue跳过本次，break跳出整个循环
