# GDB高级调试技巧

## 一、高级断点

### 1.1 条件断点

```bash
# 基本条件断点
(gdb) break 42 if x > 100
(gdb) b main.c:25 if count == 10
(gdb) b myfunction if *ptr == NULL

# 使用复杂条件
(gdb) b 50 if (x > 0) && (y < 100)
(gdb) b process if strcmp(name, "test") == 0

# 修改断点条件
(gdb) condition 1 x > 200

# 取消断点条件
(gdb) condition 1
```

### 1.2 硬件断点（Watchpoint）

```bash
# 写监视点（变量被写入时触发）
(gdb) watch x
(gdb) watch *ptr
(gdb) watch my_struct.member

# 读监视点（变量被读取时触发，需要硬件支持）
(gdb) rwatch x

# 读写监视点
(gdb) awatch x

# 查看监视点
(gdb) info watchpoints

# 示例：监视循环中的变量
(gdb) watch sum
(gdb) continue
# 每次sum改变时都会停下来
```

### 1.3 访问监视点

```bash
# 监视特定地址的访问
(gdb) watch *(int *)0x7fffffffe010

# 监视数组元素
(gdb) watch arr[5]
(gdb) watch *(arr + index)
```

## 二、捕获点（Catchpoints）

### 2.1 系统调用捕获

```bash
# 捕获所有系统调用
(gdb) catch syscall

# 捕获特定系统调用
(gdb) catch syscall open
(gdb) catch syscall read
(gdb) catch syscall write
(gdb) catch syscall mmap
(gdb) catch syscall exit_group

# 查看系统调用号
(gdb) info syscalls
```

### 2.2 异常捕获

```bash
# 捕获信号
(gdb) catch signal SIGSEGV
(gdb) catch signal SIGABRT
(gdb) catch signal SIGFPE

# 捕获C++异常（需要C++支持）
(gdb) catch throw
(gdb) catch catch
```

### 2.3 进程事件捕获

```bash
# 捕获fork
(gdb) catch fork
(gdb) catch vfork

# 捕获exec
(gdb) catch exec

# 捕获加载/卸载共享库
(gdb) catch load
(gdb) catch unload
```

## 三、多线程调试

### 3.1 线程管理

```bash
# 查看所有线程
(gdb) info threads

# 输出示例：
#   Id   Target Id         Frame
# * 1    Thread 0x7f...    worker (arg=0x0) at main.c:30
#   2    Thread 0x7f...    worker (arg=0x1) at main.c:30
#   3    Thread 0x7f...    __pthread_cond_wait at cond.c:42

# 切换到指定线程
(gdb) thread 2

# 在所有线程中执行命令
(gdb) thread apply all bt

# 在指定线程中执行命令
(gdb) thread apply 1 2 bt

# 设置线程断点
(gdb) break main.c:30 thread 2

# 锁定线程（只调试当前线程）
(gdb) set non-stop on
(gdb) set scheduler-locking on
```

### 3.2 多线程调试技巧

```bash
# 跟踪线程创建
(gdb) set follow-fork-mode child   # 跟踪子进程

# 避免线程切换干扰调试
(gdb) set scheduler-locking step   # 单步时锁定其他线程
(gdb) set scheduler-locking on     # 完全锁定

# 查看线程的调用栈
(gdb) thread apply all where
```

## 四、多进程调试

### 4.1 fork调试

```bash
# 设置follow-fork模式
(gdb) set follow-fork-mode child   # 调试子进程
(gdb) set follow-fork-mode parent  # 调试父进程（默认）

# 设置是否分离fork的子进程
(gdb) set detach-on-fork on        # 分离子进程
(gdb) set detach-on-fork off       # 不分离，都由GDB管理

# 查看所有被调试的进程
(gdb) info inferiors

# 切换进程
(gdb) inferior 2

# 同时调试父子进程
(gdb) set detach-on-fork off
(gdb) set schedule-multiple on
```

### 4.2 attach/detach进程

```bash
# 附加到运行中的进程
$ gdb -p <PID>
# 或
(gdb) attach <PID>

# 分离进程（不终止）
(gdb) detach

# 杀死被调试进程
(gdb) kill
```

## 五、反汇编与底层调试

### 5.1 反汇编

```bash
# 反汇编当前函数
(gdb) disassemble
(gdb) disas

# 反汇编指定函数
(gdb) disassemble main
(gdb) disas factorial

# 反汇编指定地址范围
(gdb) disas 0x401100,0x401200

# 使用Intel语法
(gdb) set disassembly-flavor intel
(gdb) disas main

# 混合显示源码和汇编
(gdb) disas /s main
(gdb) disas /m main       # 混合模式
```

### 5.2 寄存器操作

```bash
# 查看所有寄存器
(gdb) info registers
(gdb) i r

# 查看特定寄存器
(gdb) p $rax
(gdb) p $rip
(gdb) p $rsp
(gdb) p/x $rflags

# 修改寄存器
(gdb) set $rax = 0x1234
(gdb) set $rip = 0x401100

# 查看浮点寄存器
(gdb) info all-registers
(gdb) info float

# 查看向量寄存器
(gdb) p $xmm0
```

### 5.3 汇编级单步

```bash
# 汇编级单步
(gdb) stepi           # 或 si，执行一条汇编指令
(gdb) nexti           # 或 ni，执行一条汇编指令（不进入函数）

# 在汇编和源码模式间切换
(gdb) set step-mode on    # 强制汇编级单步
```

## 六、内存分析

### 6.1 内存查看

```bash
# 查看内存内容
(gdb) x/32xb ptr      # 32个字节，十六进制
(gdb) x/16xw ptr      # 16个4字节
(gdb) x/8xg ptr       # 8个8字节
(gdb) x/s str         # 字符串
(gdb) x/20i $pc       # 20条指令

# 查看大端/小端
(gdb) x/4xb &value    # 查看4字节的字节排列

# 搜索内存
(gdb) find /b 0x7fffffffe000, 0x7ffffffff000, 0x41, 0x42, 0x43
# 在指定范围内搜索字节序列
```

### 6.2 内存修改

```bash
# 修改内存
(gdb) set {int}0x7fffffffe010 = 42
(gdb) set {char}ptr = 'A'
(gdb) set {long}$rsp = 0x12345678

# 使用set命令修改
(gdb) set *(int *)0x7fffffffe010 = 42

# 填充内存
(gdb) set {char [10]}buffer = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
```

### 6.3 检查内存映射

```bash
# 查看内存映射
(gdb) info proc mappings

# 查看进程状态
(gdb) info proc status
(gdb) info proc exe
(gdb) info proc cmdline
```

## 七、调试信息增强

### 7.1 使用DWARF调试信息

```bash
# 编译时包含最多调试信息
gcc -g3 -gdwarf-4 program.c -o program

# -g3 包含宏定义
# -gdwarf-4 使用DWARF 4格式
```

### 7.2 宏调试

```bash
# 需要 -g3 编译

# 查看宏定义
(gdb) macro expand MAX(a, b)
(gdb) info macro MAX

# 在宏展开后查看
(gdb) macro define DEBUG_PRINT(x) printf("DEBUG: %d\n", x)
```

## 八、GDB Python扩展

### 8.1 Python脚本

```python
# gdb_script.py
import gdb

class MyCommand(gdb.Command):
    """自定义GDB命令示例"""
    def __init__(self):
        super(MyCommand, self).__init__("mycommand", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        frame = gdb.selected_frame()
        pc = frame.pc()
        print(f"当前PC: 0x{pc:x}")

MyCommand()
```

```bash
# 加载Python脚本
(gdb) source gdb_script.py
(gdb) mycommand
```

### 8.2 Pretty Printer

```python
# 自定义类型打印
class MyStructPrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        return f"MyStruct(x={self.val['x']}, y={self.val['y']})"
```

## 九、Core Dump分析

### 9.1 生成Core Dump

```bash
# 允许生成core dump
ulimit -c unlimited

# 运行程序（崩溃时生成core文件）
./program
# 生成 core 或 core.<PID>

# 指定core文件名
echo "core.%e.%p" > /proc/sys/kernel/core_pattern
```

### 9.2 分析Core Dump

```bash
# 加载core文件
gdb program core

# 或
(gdb) core-file core

# 查看崩溃位置
(gdb) where
(gdb) bt full

# 查看崩溃时的变量
(gdb) info locals
(gdb) info registers

# 查看崩溃的信号
(gdb) info signals
```

### 9.3 常见崩溃分析

```bash
# Segmentation Fault (SIGSEGV)
# 通常是空指针或越界访问
(gdb) bt
(gdb) print *ptr        # 检查指针
(gdb) print ptr         # 检查地址

# Abort (SIGABRT)
# 通常是assert失败或double free
(gdb) bt
(gdb) info signals

# Floating Point Exception (SIGFPE)
# 通常是除零错误
(gdb) bt
(gdb) frame 1
(gdb) print divisor
```

## 十、TUI模式

### 10.1 TUI界面

```bash
# 启动TUI模式
gdb -tui program

# 在GDB中切换TUI
(gdb) tui enable
(gdb) tui disable
(gdb) tui toggle

# TUI布局
(gdb) layout src        # 源码窗口
(gdb) layout asm        # 汇编窗口
(gdb) layout split      # 源码+汇编
(gdb) layout regs       # 寄存器窗口

# TUI快捷键
# Ctrl+x a    切换TUI模式
# Ctrl+x o    切换焦点窗口
# Ctrl+x 1    单窗口模式
# Ctrl+x 2    双窗口模式
# Ctrl+l      刷新屏幕
```

### 10.2 Dashboard

```bash
# 使用dashboard插件增强GDB界面
# https://github.com/cyrus-and/gdb-dashboard

# 安装
wget -P ~ https://git.io/.gdbinit
# 或
curl -s https://api.github.com/repos/cyrus-and/gdb-dashboard/releases/latest \
  | grep browser_download_url | cut -d '"' -f 4 | wget -qi -
```

## 十一、调试优化代码

### 11.1 优化对调试的影响

```bash
# 编译优化后可能的问题：
# 1. 变量被优化掉（optimized out）
# 2. 执行顺序与源码不一致
# 3. 内联函数无法单步进入

# 解决方案：
gcc -Og -g program.c -o program
# -Og 是对调试友好的优化级别
```

### 11.2 处理优化后的调试

```bash
# 查看变量是否被优化
(gdb) p x
# 输出: <optimized out>

# 强制查看值
(gdb) frame
(gdb) info locals

# 使用-Og编译可避免大部分问题
gcc -Og -g program.c -o program
```

## 十二、GDB配置文件

### 12.1 .gdbinit文件

```bash
# ~/.gdbinit（全局配置）
set print pretty on
set print array on
set print array-indexes on
set demangle-style gnu-v3
set history save on
set history size 10000

# 自定义命令
define ll
    info locals
end

define la
    info args
end

define btall
    thread apply all bt
end
```

### 12.2 项目级配置

```bash
# 项目目录下的 .gdbinit
# 需要在 ~/.gdbinit 中添加：
set auto-load safe-path /

# 项目 .gdbinit
set args --config test.ini
break main
run
```

## 十三、关键要点

> **重要提示**：
> 1. `watch` 命令可以在变量值改变时自动中断，用于追踪内存修改
> 2. 条件断点：`break location if condition`
> 3. 多线程调试使用 `info threads` 和 `thread N`
> 4. `disassemble` 可以查看汇编代码，`stepi/nexti` 进行汇编级单步
> 5. `catch syscall` 可以捕获系统调用
> 6. Core Dump分析是排查崩溃问题的关键工具
> 7. TUI模式提供类似IDE的调试界面
> 8. 使用 `-Og -g` 编译以获得最佳调试体验
> 9. `.gdbinit` 文件可以保存常用调试配置
> 10. `x/NFU addr` 命令可以灵活查看内存内容
