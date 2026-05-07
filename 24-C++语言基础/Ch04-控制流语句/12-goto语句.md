# goto语句

## 一、概念说明

`goto`无条件跳转到同一函数内的标号处。它被认为是有害的，会破坏结构化编程，但在极少数场景下（如跳出多层嵌套循环、错误清理）有其合理性。

## 二、具体用法

### 2.1 基本语法

```cpp
#include <iostream>
using namespace std;

int main() {
    int i = 0;

start:  // 标号
    if (i >= 3) goto end;
    cout << "i = " << i << endl;
    i++;
    goto start;

end:
    cout << "结束" << endl;

    return 0;
}
```

输出：
```
i = 0
i = 1
i = 2
结束
```

### 2.2 合理使用场景

```cpp
#include <iostream>
using namespace std;

int main() {
    // 场景：跳出多层嵌套循环
    // 比使用标志变量更简洁
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (i * j > 20) {
                cout << "找到: i=" << i << ", j=" << j << endl;
                goto found;
            }
        }
    }
found:

    // 场景：错误清理（C风格，现代C++用RAII）
    // FILE* f = fopen("file.txt", "r");
    // if (!f) goto error;
    // // ... 处理 ...
    // fclose(f);
    // error:
    //     cerr << "错误" << endl;

    return 0;
}
```

输出：
```
找到: i=3, j=7
```

## 三、注意事项与常见陷阱

1. **不要使用goto**：除极少数情况外，用break/continue/return替代
2. **不能跳过变量初始化**：`goto`跳过变量声明是编译错误
3. **只在函数内有效**：不能跳转到其他函数
4. **破坏可读性**：goto使控制流难以追踪
5. **替代方案**：用函数提取、RAII、标志变量等替代goto
