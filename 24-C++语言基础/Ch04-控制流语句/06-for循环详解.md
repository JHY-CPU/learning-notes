# for循环详解

## 一、概念说明

`for`循环是最常用的循环结构，适合**已知迭代次数**的场景。它包含初始化、条件判断和迭代三个部分。

## 二、具体用法

### 2.1 基本for循环

```cpp
#include <iostream>
using namespace std;

int main() {
    // 标准for循环
    // for (初始化; 条件; 迭代) { 循环体 }
    for (int i = 0; i < 5; i++) {
        cout << i << " ";
    }
    cout << endl;

    // 多变量
    for (int i = 0, j = 10; i < j; i++, j--) {
        cout << "(" << i << "," << j << ") ";
    }
    cout << endl;

    // 省略部分（不推荐，但合法）
    int k = 0;
    for (; k < 3; ) {
        cout << k << " ";
        k++;
    }
    cout << endl;

    // 无限循环
    int count = 0;
    for (;;) {
        if (count >= 3) break;
        cout << "循环" << count << " ";
        count++;
    }
    cout << endl;

    return 0;
}
```

输出：
```
0 1 2 3 4
(0,10) (1,9) (2,8) (3,7) (4,6)
0 1 2
循环0 循环1 循环2
```

### 2.2 for循环执行流程

```cpp
#include <iostream>
using namespace std;

int main() {
    // 执行流程：
    // 1. 初始化（只执行一次）
    // 2. 条件判断（每次迭代前）
    // 3. 循环体（条件为true时）
    // 4. 迭代表达式（每次迭代后）
    // 5. 回到步骤2

    for (int i = 0; i < 3; i++) {
        cout << "循环体 i=" << i << endl;
    }
    // 执行顺序：
    // i=0 → 判断i<3(true) → 循环体 → i++ →
    // i=1 → 判断i<3(true) → 循环体 → i++ →
    // i=2 → 判断i<3(true) → 循环体 → i++ →
    // i=3 → 判断i<3(false) → 退出

    return 0;
}
```

输出：
```
循环体 i=0
循环体 i=1
循环体 i=2
```

## 三、注意事项与常见陷阱

1. **差一错误**：`for (i = 0; i <= n; i++)`执行n+1次，`i < n`执行n次
2. **循环变量类型**：与容器大小比较时注意`int`和`size_t`（无符号）的区别
3. **迭代部分可以为空或多个**：`for (;cond; a++, b++, c++)`是合法的
4. **初始化部分声明的变量只在for内可见**（C++11起）
5. **避免在循环体内修改循环变量**：可能导致意外行为
