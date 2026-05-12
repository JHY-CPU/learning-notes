# continue语句

## 一、概念说明

`continue`语句（C++11标准 §6.6.2）跳过当前迭代的**剩余部分**，直接进入下一次迭代。它只影响最近的一层循环，不终止整个循环。

与 `break` 的区别：

| 语句 | 作用 | 影响范围 |
|------|------|---------|
| `continue` | 跳过本次迭代剩余部分 | 当前循环 |
| `break` | 跳出整个循环 | 当前循环 |
| `return` | 结束整个函数 | 当前函数 |

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    // 跳过偶数，只打印奇数
    for (int i = 1; i <= 10; i++) {
        if (i % 2 == 0) continue;
        cout << i << " ";
    }
    cout << endl;  // 1 3 5 7 9

    // 跳过特定值
    vector<int> nums = {1, -2, 3, -4, 5, 0, 7};
    int sum = 0;
    for (int n : nums) {
        if (n <= 0) continue;  // 跳过非正数
        sum += n;
    }
    cout << "正数之和: " << sum << endl;  // 16

    return 0;
}
```

### 2.2 在不同循环类型中的行为差异

```cpp
#include <iostream>
using namespace std;

int main() {
    // for循环：continue后执行 i++（第三表达式）
    cout << "for: ";
    for (int i = 0; i < 5; i++) {
        if (i == 2) continue;  // 跳过i==2
        cout << i << " ";      // 0 1 3 4
    }
    cout << endl;

    // while循环：continue后回到条件判断，可能跳过更新语句！
    cout << "while: ";
    int j = 0;
    while (j < 5) {
        j++;
        if (j == 3) continue;  // j++在continue之前，安全
        cout << j << " ";      // 1 2 4 5
    }
    cout << endl;

    // 危险示例：continue跳过更新导致死循环
    /*
    int k = 0;
    while (k < 5) {
        if (k == 0) continue;  // 死循环！k永远是0
        k++;
    }
    */

    // do-while：类似while，回到while条件判断
    cout << "do-while: ";
    int m = 0;
    do {
        m++;
        if (m == 3) continue;
        cout << m << " ";      // 1 2 4 5
    } while (m < 5);
    cout << endl;

    return 0;
}
```

### 2.3 嵌套循环中的continue

```cpp
#include <iostream>
using namespace std;

int main() {
    // continue只影响内层循环，不影响外层
    for (int i = 0; i < 3; i++) {
        cout << "行" << i << ": ";
        for (int j = 0; j < 5; j++) {
            if (j == 2) continue;  // 跳过j==2
            cout << j << " ";
        }
        cout << endl;
    }

    // 如果需要跳过外层迭代，使用标志变量或goto
    for (int i = 0; i < 3; i++) {
        bool skipRow = false;
        for (int j = 0; j < 5; j++) {
            if (i == 1 && j == 2) {
                skipRow = true;
                break;  // 跳出内层
            }
            cout << "(" << i << "," << j << ") ";
        }
        if (skipRow) {
            cout << " [跳过第1行剩余]";
        }
        cout << endl;
    }

    return 0;
}
```

### 2.4 C++17 结构化绑定中的continue

```cpp
#include <iostream>
#include <map>
#include <string>
using namespace std;

int main() {
    map<string, int> scores = {{"Alice", 95}, {"Bob", 60}, {"Carol", 78}};

    // 只处理及格的学生
    for (const auto& [name, score] : scores) {
        if (score < 70) continue;
        cout << name << " 及格，分数: " << score << endl;
    }

    return 0;
}
```

## 三、最佳实践

1. **优先将continue条件放在循环体开头**：使逻辑更清晰
2. **避免在循环中间使用continue**：降低可读性
3. **检查while循环中的更新语句位置**：确保continue不会跳过更新
4. **嵌套循环中考虑用函数封装**：用return替代多层continue

## 四、注意事项与常见陷阱

1. **continue后代码不执行**：当前迭代中continue之后的语句被跳过
2. **while循环注意迭代更新**：continue可能跳过`i++`导致死循环，将更新放在continue之前可避免
3. **只影响一层循环**：嵌套循环中continue不影响外层
4. **可读性**：避免在循环开头连续多个continue，考虑用if重写条件
5. **与break区别**：continue跳过本次，break跳出整个循环；在switch中不能使用continue
6. **continue不能用于switch语句**：只能用在for/while/do-while循环中
