# goto语句

## 一、概念说明

`goto`语句（C++11标准 §6.6.4）实现无条件跳转到同一函数内的标号处。自Dijkstra在1968年发表《Go To Statement Considered Harmful》以来，goto一直被视为破坏结构化编程的元凶，但在**极少数场景**下（跳出多层嵌套循环、集中式错误清理）仍有一定合理性。

现代C++中goto的合理替代方案：

| goto用途 | 现代替代 |
|---------|---------|
| 跳出多层循环 | 提取函数 + return、异常、标志变量 |
| 错误清理 | RAII、异常、`std::unique_ptr`自定义删除器 |
| 状态机 | 状态模式、协程（C++20） |

## 二、具体用法

### 2.1 基本语法

```cpp
#include <iostream>
using namespace std;

int main() {
    int i = 0;

start:  // 标号（函数内唯一标识）
    if (i >= 3) goto end;
    cout << "i = " << i << endl;
    i++;
    goto start;      // 向前跳转

end:
    cout << "结束" << endl;

    return 0;
}
// 输出：
// i = 0
// i = 1
// i = 2
// 结束
```

### 2.2 跳出多层嵌套循环

```cpp
#include <iostream>
using namespace std;

int main() {
    // 场景1：用goto跳出多层循环（最简洁的方式）
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i * j * k > 100) {
                    cout << "找到: " << i << "," << j << "," << k << endl;
                    goto found;
                }
            }
        }
    }
found:
    cout << "搜索结束" << endl;

    // 场景2：不用goto的替代（C++17 结构化绑定不好用时）
    bool done = false;
    for (int i = 0; i < 10 && !done; i++) {
        for (int j = 0; j < 10 && !done; j++) {
            if (i * j > 20) {
                cout << "找到: " << i << "," << j << endl;
                done = true;
            }
        }
    }

    return 0;
}
```

### 2.3 错误清理模式

```cpp
#include <iostream>
#include <cstring>
using namespace std;

// C风格的错误清理（现代C++用RAII替代）
int processFile(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        cerr << "无法打开文件" << endl;
        goto error;
    }

    char* buffer = new char[1024];
    if (!fgets(buffer, 1024, f)) {
        cerr << "读取失败" << endl;
        goto cleanup_buffer;
    }

    cout << "读取: " << buffer << endl;

cleanup_buffer:
    delete[] buffer;
cleanup_file:
    fclose(f);
error:
    return 0;
}

// 现代C++等价写法（推荐）
/*
#include <memory>
#include <fstream>
int processFileModern(const char* filename) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen(filename, "r"), &fclose);
    if (!file) return -1;

    auto buffer = std::make_unique<char[]>(1024);
    if (!fgets(buffer.get(), 1024, file.get())) return -1;

    cout << "读取: " << buffer.get() << endl;
    return 0;
    // 自动清理：unique_ptr析构释放内存和关闭文件
}
*/
```

## 三、goto的限制

```cpp
int main() {
    int x = 10;

    // goto不能跳过变量的初始化（编译错误）
    // if (x > 5) goto skip;
    // int y = 20;  // 编译错误：跳过初始化
    // skip:
    //     cout << y << endl;

    // 但跳过声明是允许的（只要不初始化）
    if (x > 5) goto skip2;
    int y;       // 声明OK
    y = 20;      // 初始化单独写
skip2:
    y = 30;
    cout << y << endl;

    // goto不能跳转到其他函数
    // 其他函数处的标号对当前函数不可见

    return 0;
}
```

## 四、注意事项与常见陷阱

1. **不要使用goto**：除极少数情况外，用break/continue/return/函数替代
2. **不能跳过变量初始化**：`goto`跳过有初始化的变量声明是编译错误，但可以跳过未初始化的声明
3. **只在函数内有效**：不能跳转到其他函数，标号的作用域限定在当前函数
4. **破坏可读性**：goto使控制流难以追踪，增加维护成本
5. **替代方案优先级**：函数提取 > RAII > 标志变量 > goto
6. **向后跳转（goto到前面的标号）可能导致死循环**：逻辑上等同于循环但更难理解
7. **goto在switch中的特殊用法**：可以跳到switch内外，但需注意作用域
