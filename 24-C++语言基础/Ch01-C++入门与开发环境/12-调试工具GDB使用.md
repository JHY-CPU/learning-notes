# 调试工具GDB使用

## 一、概念说明

GDB（GNU Debugger）是Linux/Unix下最常用的C/C++调试器。它允许你观察程序内部状态、设置断点、单步执行，是排查bug的利器。

## 二、具体用法

### 2.1 准备调试程序

```cpp
// debug_demo.cpp —— 用于演示调试的程序
#include <iostream>
using namespace std;

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int sumArray(int arr[], int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // 故意的bug：<=应为<
        sum += arr[i];
    }
    return sum;
}

int main() {
    cout << "5的阶乘: " << factorial(5) << endl;

    int nums[] = {1, 2, 3, 4, 5};
    cout << "数组求和: " << sumArray(nums, 5) << endl;

    return 0;
}
```

```bash
# 编译时必须加 -g 选项生成调试信息
g++ -g -O0 -std=c++17 debug_demo.cpp -o debug_demo
```

### 2.2 启动GDB

```bash
# 方式一：直接启动
gdb ./debug_demo

# 方式二：带核心转储文件
gdb ./debug_demo core

# 方式三：附加到运行中的进程
gdb -p <pid>
```

### 2.3 常用GDB命令

```bash
# ===== 断点管理 =====
(gdb) break main              # 在main函数入口设断点
(gdb) break debug_demo.cpp:10 # 在第10行设断点
(gdb) break sumArray          # 在函数入口设断点
(gdb) info breakpoints        # 列出所有断点
(gdb) delete 2                # 删除2号断点
(gdb) disable 1               # 禁用1号断点
(gdb) enable 1                # 启用1号断点

# ===== 执行控制 =====
(gdb) run                     # 启动程序
(gdb) run arg1 arg2           # 带参数启动
(gdb) next                    # 单步执行（不进入函数）
(gdb) step                    # 单步执行（进入函数）
(gdb) continue                # 继续执行到下一个断点
(gdb) finish                  # 执行到当前函数返回
(gdb) until 20                # 执行到第20行

# ===== 查看信息 =====
(gdb) print n                 # 打印变量n的值
(gdb) print factorial(n)      # 调用函数并打印结果
(gdb) print arr[0]@5          # 打印数组arr的前5个元素
(gdb) info locals             # 显示所有局部变量
(gdb) info args               # 显示函数参数
(gdb) backtrace               # 显示调用栈
(gdb) frame 2                 # 切换到第2个栈帧
(gdb) list                    # 显示当前位置的源代码
(gdb) display sum             # 每步执行后自动显示sum的值

# ===== 修改与控制 =====
(gdb) set var n = 10          # 修改变量值
(gdb) watch sum               # 设置观察点，sum变化时中断
(gdb) quit                    # 退出GDB
```

### 2.4 调试会话示例

```
$ gdb ./debug_demo
(gdb) break sumArray
Breakpoint 1 at 0x1149: file debug_demo.cpp, line 12.
(gdb) run
Starting program: ./debug_demo
5的阶乘: 120

Breakpoint 1, sumArray (arr=0x7fffffffe050, size=5) at debug_demo.cpp:12
12	    int sum = 0;
(gdb) next
13	    for (int i = 0; i <= size; i++) {
(gdb) next
14	        sum += arr[i];
(gdb) print i
$1 = 0
(gdb) continue
Continuing.
数组求和: 15

Program received signal SIGSEGV, Segmentation fault.
0x00005555555551a2 in sumArray (arr=0x7fffffffe050, size=5) at debug_demo.cpp:14
14	        sum += arr[i];
(gdb) print i
$2 = 5
(gdb) quit
```

## 三、注意事项与常见陷阱

1. **必须用-g编译**：没有`-g`选项GDB无法显示源代码和变量名
2. **-O0配合调试**：优化后的代码执行顺序可能与源码不一致，调试时建议用`-O0`或`-Og`
3. **TUI模式**：`gdb -tui`或在GDB中按`Ctrl+X+A`可以开启文本界面，同时看源码
4. **GDB的GUI前端**：DDD、VS Code、CLion等IDE提供了图形化的GDB前端
5. **调试核心转储**：程序崩溃时，用`ulimit -c unlimited`启用核心转储，然后用GDB分析
