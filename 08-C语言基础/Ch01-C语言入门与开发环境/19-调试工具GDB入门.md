# 调试工具GDB入门

## 一、GDB简介

GDB（GNU Debugger）是GNU项目的标准调试器，用于调试C/C++程序。它支持设置断点、单步执行、查看变量、检查内存等操作。

### 1.1 编译要求

```bash
# 必须使用 -g 选项编译，生成调试信息
gcc -g program.c -o program

# 推荐的调试编译选项
gcc -Wall -Wextra -g -O0 program.c -o program

# -g  生成调试信息
# -O0 禁用优化（避免调试时代码与源码不一致）
```

## 二、启动GDB

### 2.1 基本启动方式

```bash
# 方式1：直接启动并加载程序
gdb program

# 方式2：启动后加载程序
gdb
(gdb) file program

# 方式3：启动并传入参数
gdb --args program arg1 arg2

# 方式4：调试运行中的进程
gdb program <PID>
gdb -p <PID>

# 方式5：分析core dump
gdb program core
```

### 2.2 退出GDB

```
(gdb) quit        # 或 q
(gdb) q
```

## 三、运行程序

### 3.1 启动执行

```bash
# 不带参数运行
(gdb) run           # 或 r

# 带参数运行
(gdb) run arg1 arg2

# 设置命令行参数后运行
(gdb) set args arg1 arg2
(gdb) run

# 查看当前参数
(gdb) show args

# 使用shell命令
(gdb) shell ls
(gdb) shell clear
```

### 3.2 程序输入输出

```bash
# 重定向输入
(gdb) run < input.txt

# 重定向输出
(gdb) run > output.txt
```

## 四、断点管理

### 4.1 设置断点

```bash
# 在函数入口设断点
(gdb) break main
(gdb) b main

# 在指定文件的指定行设断点
(gdb) break filename:linenum
(gdb) b main.c:25

# 在当前文件指定行设断点
(gdb) break 42
(gdb) b 42

# 在函数入口处（多文件）
(gdb) break filename:function
(gdb) break utils.c:helper

# 条件断点
(gdb) break 42 if x > 100
(gdb) b main.c:25 if count == 10
```

### 4.2 查看断点

```bash
# 列出所有断点
(gdb) info breakpoints    # 或 i b
(gdb) i breakpoints
```

输出示例：
```
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x0000000000401136 in main at main.c:10
        breakpoint already hit 2 times
2       breakpoint     keep y   0x0000000000401150 in main at main.c:25
        stop only if x > 100
```

### 4.3 管理断点

```bash
# 删除断点
(gdb) delete 1          # 删除1号断点
(gdb) delete 1 3 5      # 删除多个
(gdb) delete breakpoints # 删除所有断点
(gdb) d 1               # 简写

# 禁用断点
(gdb) disable 1         # 禁用1号断点
(gdb) disable           # 禁用所有

# 启用断点
(gdb) enable 1          # 启用1号断点
(gdb) enable            # 启用所有

# 一次性断点（命中后自动删除）
(gdb) tbreak main       # 临时断点
(gdb) tb 42

# 忽略断点前N次命中
(gdb) ignore 1 10       # 忽略1号断点的前10次命中
```

## 五、单步执行

### 5.1 执行控制

```bash
# 继续执行到下一个断点
(gdb) continue           # 或 c

# 单步执行（Step Over，不进入函数）
(gdb) next               # 或 n
(gdb) next 5             # 执行5步

# 单步执行（Step Into，进入函数）
(gdb) step               # 或 s
(gdb) step 3             # 执行3步

# 执行到函数返回（Step Out）
(gdb) finish             # 或 fin

# 执行到指定行
(gdb) until 50           # 或 u 50
(gdb) until main.c:50

# 执行到当前函数结束
(gdb) finish
```

### 5.2 执行控制总结

```
┌──────────────────────────────────────────────────────┐
│ 命令      │ 快捷键 │ 行为                            │
├───────────┼────────┼─────────────────────────────────┤
│ run       │ r      │ 从头开始执行程序                 │
│ continue  │ c      │ 继续执行到下一个断点              │
│ next      │ n      │ 执行下一行（不进入函数）          │
│ step      │ s      │ 执行下一行（进入函数）            │
│ finish    │ fin    │ 执行到当前函数返回                │
│ until     │ u      │ 执行到指定行/函数结束             │
│ return    │        │ 立即从当前函数返回                │
│ quit      │ q      │ 退出GDB                         │
└──────────────────────────────────────────────────────┘
```

## 六、查看数据

### 6.1 查看变量

```bash
# 打印变量值
(gdb) print x            # 或 p x
(gdb) print/x x          # 十六进制
(gdb) print/t x          # 二进制
(gdb) print/c x          # 字符
(gdb) print/f x          # 浮点数

# 格式化打印
(gdb) p/d x              # 十进制
(gdb) p/x x              # 十六进制
(gdb) p/o x              # 八进制
(gdb) p/t x              # 二进制
(gdb) p/a x              # 地址
(gdb) p/c x              # 字符
(gdb) p/f x              # 浮点数
(gdb) p/s str            # 字符串

# 修改变量值
(gdb) set var x = 42
(gdb) set var name = "test"

# 自动显示变量（每步执行后自动打印）
(gdb) display x
(gdb) display/x x
(gdb) display *ptr@10    # 显示指针指向的10个元素

# 取消自动显示
(gdb) undisplay 1
(gdb) delete display

# 查看所有自动显示
(gdb) info display
```

### 6.2 查看数据结构

```bash
# 查看数组
(gdb) p arr              # 打印整个数组
(gdb) p arr[0]@10        # 打印arr[0]开始的10个元素
(gdb) p arr[2]@5         # 打印arr[2]开始的5个元素

# 查看结构体
(gdb) p my_struct        # 打印整个结构体
(gdb) p my_struct.member # 打印结构体成员
(gdb) p *ptr             # 解引用指针

# 漂亮打印（pretty print）
(gdb) set print pretty on
(gdb) p my_struct

# 查看动态数组
(gdb) set print elements 0    # 不限制打印元素数量
(gdb) p *arr@count            # 打印count个元素

# 查看联合体
(gdb) p my_union
(gdb) p my_union.int_member
```

### 6.3 查看内存

```bash
# 查看内存（examine）
(gdb) x/10xw ptr         # 查看10个4字节的十六进制值
(gdb) x/20xb ptr         # 查看20个字节
(gdb) x/5dw ptr          # 查看5个4字节的十进制值
(gdb) x/s str_ptr        # 查看字符串
(gdb) x/i $pc            # 查看当前指令

# x命令格式：x/NFU addr
# N: 元素数量
# F: 格式 (x=hex, d=decimal, t=binary, c=char, s=string, i=instruction)
# U: 单位 (b=byte, h=halfword, w=word, g=giant/8bytes)
```

## 七、查看程序状态

### 7.1 调用栈

```bash
# 查看调用栈
(gdb) backtrace           # 或 bt
(gdb) where               # 同bt
(gdb) info stack

# 完整调用栈
(gdb) backtrace full

# 切换栈帧
(gdb) frame 2             # 或 f 2
(gdb) up                  # 上移一帧
(gdb) down                # 下移一帧

# 查看当前帧信息
(gdb) info frame
(gdb) info locals         # 当前函数局部变量
(gdb) info args           # 当前函数参数
```

### 7.2 查看源代码

```bash
# 列出源代码
(gdb) list               # 或 l，显示当前位置代码
(gdb) list 10,20         # 显示10-20行
(gdb) list main          # 显示main函数
(gdb) list filename:25   # 显示指定文件的25行

# 设置列表行数
(gdb) set listsize 20

# 显示当前位置
(gdb) info source
(gdb) info line
```

### 7.3 查看寄存器

```bash
# 查看所有寄存器
(gdb) info registers      # 或 i r
(gdb) info all-registers

# 查看特定寄存器
(gdb) p $rax
(gdb) p $pc
(gdb) p $rsp
```

## 八、实用技巧

### 8.1 捕获点（Catchpoints）

```bash
# 捕获异常
(gdb) catch throw

# 捕获系统调用
(gdb) catch syscall open

# 捕获fork/exec
(gdb) catch fork
(gdb) catch exec
```

### 8.2 信号处理

```bash
# 查看信号设置
(gdb) info signals

# 忽略信号
(gdb) handle SIGINT nostop noprint

# 捕获信号
(gdb) handle SIGSEGV stop print
```

### 8.3 命令历史和脚本

```bash
# 查看命令历史
(gdb) show commands

# 执行GDB命令脚本
(gdb) source gdb_commands.txt

# 保存命令历史
(gdb) set history save on
(gdb) set history filename ~/.gdb_history
```

## 九、调试示例

### 9.1 示例程序

```c
/* debug_example.c */
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main(void) {
    int nums[] = {1, 2, 3, 4, 5};
    int sum = 0;

    for (int i = 0; i < 5; i++) {
        sum += nums[i];
    }

    printf("Sum = %d\n", sum);
    printf("5! = %d\n", factorial(5));

    return 0;
}
```

### 9.2 调试会话示例

```bash
$ gcc -g -O0 debug_example.c -o debug_example
$ gdb ./debug_example

(gdb) break main
(gdb) break factorial
(gdb) run
(gdb) next                # 执行到 sum += nums[i]
(gdb) print i
(gdb) print nums[i]
(gdb) display sum
(gdb) continue
(gdb) step                # 进入factorial函数
(gdb) where               # 查看调用栈
(gdb) print n
(gdb) finish              # 执行到factorial返回
(gdb) continue
```

## 十、GDB与IDE

### 10.1 VSCode配置

```json
// launch.json
{
    "version": "0.2.0",
    "configurations": [{
        "name": "GDB Debug",
        "type": "cppdbg",
        "request": "launch",
        "program": "${workspaceFolder}/build/program",
        "MIMode": "gdb",
        "miDebuggerPath": "/usr/bin/gdb",
        "setupCommands": [{
            "text": "-enable-pretty-printing"
        }]
    }]
}
```

### 10.2 CLion

CLion内置GDB集成，通过图形界面操作断点、变量、调用栈等。

## 十一、关键要点

> **重要提示**：
> 1. 使用 `-g -O0` 编译以获得最佳调试体验
> 2. `break` 设断点，`run` 启动程序，`continue` 继续执行
> 3. `next` 不进入函数，`step` 进入函数，`finish` 返回
> 4. `print` 查看变量，`display` 自动显示变量
> 5. `backtrace` 查看调用栈，`frame` 切换栈帧
> 6. `x/NFU addr` 查看内存内容
> 7. `set var x = 42` 可以在运行时修改变量值
> 8. 条件断点：`break location if condition`
> 9. `info locals` 查看所有局部变量
> 10. 调试是理解程序运行过程的最佳方式
