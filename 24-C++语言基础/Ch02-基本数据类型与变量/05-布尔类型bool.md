# 布尔类型 bool

## 一、概念说明

`bool`类型只有两个值：`true`（真）和`false`（假）。布尔类型在条件判断、逻辑运算和控制流中扮演核心角色。虽然只占1字节，但布尔值通常由CPU的条件标志位来高效处理。

## 二、具体用法

### 2.1 基本使用

```cpp
#include <iostream>
using namespace std;

int main() {
    // 布尔变量
    bool isActive = true;
    bool isComplete = false;

    cout << "isActive: " << isActive << endl;      // 输出1
    cout << "isComplete: " << isComplete << endl;  // 输出0

    // 比较运算的结果是bool
    bool result = (10 > 5);
    cout << "10 > 5: " << result << endl;

    // boolalpha：以true/false输出
    cout << boolalpha;
    cout << "isActive: " << isActive << endl;
    cout << "isComplete: " << isComplete << endl;

    return 0;
}
```

输出：
```
isActive: 1
isComplete: 0
10 > 5: 1
isActive: true
isComplete: false
```

### 2.2 隐式转换

```cpp
#include <iostream>
using namespace std;

int main() {
    // 其他类型转bool：0为false，非0为true
    bool fromInt1 = 0;       // false
    bool fromInt2 = 42;      // true
    bool fromDouble = 0.0;   // false
    bool fromDouble2 = 3.14; // true
    bool fromNull = nullptr; // false

    cout << boolalpha;
    cout << "0 -> " << fromInt1 << endl;
    cout << "42 -> " << fromInt2 << endl;
    cout << "0.0 -> " << fromDouble << endl;
    cout << "3.14 -> " << fromDouble2 << endl;
    cout << "nullptr -> " << fromNull << endl;

    // bool转其他类型：false->0, true->1
    int i = true;
    double d = false;
    cout << "true -> int: " << i << endl;
    cout << "false -> double: " << d << endl;

    return 0;
}
```

输出：
```
0 -> false
42 -> true
0.0 -> false
3.14 -> true
nullptr -> false
true -> int: 1
false -> double: 0
```

### 2.3 布尔运算

```cpp
#include <iostream>
using namespace std;

int main() {
    bool a = true, b = false;

    // 逻辑与
    cout << boolalpha;
    cout << "true && false = " << (a && b) << endl;   // false

    // 逻辑或
    cout << "true || false = " << (a || b) << endl;   // true

    // 逻辑非
    cout << "!true = " << (!a) << endl;               // false
    cout << "!false = " << (!b) << endl;              // true

    // 实际应用：条件判断
    int age = 20;
    bool hasID = true;
    bool canEnter = (age >= 18) && hasID;
    cout << "可以进入: " << canEnter << endl;

    return 0;
}
```

输出：
```
true && false = false
true || false = true
!true = false
!false = false
可以进入: true
```

### 2.4 bool在控制流中的应用

```cpp
#include <iostream>
using namespace std;

bool isEven(int n) {
    return n % 2 == 0;
}

bool isInRange(int value, int min, int max) {
    return value >= min && value <= max;
}

int main() {
    // bool作为if条件
    int num = 7;
    if (isEven(num)) {
        cout << num << " 是偶数" << endl;
    } else {
        cout << num << " 是奇数" << endl;
    }

    // bool作为循环条件
    bool running = true;
    int count = 0;
    while (running) {
        count++;
        if (count >= 3) {
            running = false;
        }
    }
    cout << "循环了 " << count << " 次" << endl;

    // 条件运算符
    int score = 85;
    string grade = (score >= 60) ? "及格" : "不及格";
    cout << "成绩: " << grade << endl;

    return 0;
}
```

输出：
```
7 是奇数
循环了 3 次
成绩: 及格
```

## 三、注意事项与常见陷阱

1. **bool只有true和false**：不要给bool赋值`bool b = 2;`，虽然编译通过但语义不清
2. **bool大小是1字节**：`sizeof(bool)`结果是1，但不要假设bool数组中每个元素占1位
3. **不要用==比较bool**：`if (x == true)`应写成`if (x)`，`if (x == false)`应写成`if (!x)`
4. **整数和bool混用**：避免`int x = (a > b) + (c > d)`这种写法，可读性差
5. **指针转bool**：空指针转为false，非空指针转为true，常用于检查指针有效性
