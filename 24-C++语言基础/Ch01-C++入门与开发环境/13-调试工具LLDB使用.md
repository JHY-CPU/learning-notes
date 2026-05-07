# 调试工具LLDB使用

## 一、概念说明

LLDB是LLVM项目的调试器，是macOS上的默认调试器，也可在Linux和Windows上使用。LLDB命令与GDB类似但语法略有不同，对C++的模板和STL容器有更好的支持。

## 二、具体用法

### 2.1 启动LLDB

```bash
# 编译带调试信息的程序
clang++ -g -O0 -std=c++17 demo.cpp -o demo

# 启动LLDB
lldb ./demo

# 附加到运行中的进程
lldb -p <pid>
```

### 2.2 常用LLDB命令

```bash
# ===== 断点管理 =====
(lldb) breakpoint set -n main            # 在main函数设断点
(lldb) breakpoint set -f demo.cpp -l 15  # 在demo.cpp第15行设断点
(lldb) breakpoint set -n MyClass::func   # 在类成员函数设断点
(lldb) breakpoint list                    # 列出所有断点
(lldb) breakpoint delete 1               # 删除1号断点

# 简写形式
(lldb) b main                            # 同 breakpoint set -n main
(lldb) b demo.cpp:15                     # 同 breakpoint set -f demo.cpp -l 15

# ===== 执行控制 =====
(lldb) run                               # 启动程序
(lldb) run arg1 arg2                     # 带参数启动
(lldb) next                              # 单步（不进入函数），简写 n
(lldb) step                              # 单步（进入函数），简写 s
(lldb) continue                          # 继续执行，简写 c
(lldb) finish                            # 执行到函数返回
(lldb) thread until 20                   # 执行到第20行

# ===== 查看信息 =====
(lldb) frame variable                    # 显示当前栈帧所有变量
(lldb) frame variable sum                # 显示变量sum
(lldb) p sum                             # 打印表达式结果
(lldb) p/x 255                           # 以十六进制打印
(lldb) p arr[0]@5                        # 打印数组前5个元素
(lldb) expression sum = 100              # 修改变量值
(lldb) bt                                # 显示调用栈（backtrace）
(lldb) frame select 2                    # 切换到第2个栈帧
(lldb) target modules list               # 列出已加载的模块

# ===== 显示源代码 =====
(lldb) source list                       # 显示当前位置的源代码
(lldb) source list -n main               # 显示main函数的源代码
```

### 2.3 调试STL容器

```cpp
// debug_stl.cpp
#include <iostream>
#include <vector>
#include <string>
#include <map>
using namespace std;

int main() {
    vector<int> nums = {10, 20, 30, 40, 50};
    map<string, int> scores = {{"Alice", 95}, {"Bob", 87}};

    for (size_t i = 0; i < nums.size(); i++) {
        nums[i] *= 2;
    }

    cout << "处理完成" << endl;
    return 0;
}
```

```bash
(lldb) b 12
(lldb) run
(lldb) p nums           # 查看vector内容
(lldb) p scores         # 查看map内容
(lldb) p nums[0]        # 查看单个元素
(lldb) frame variable    # 查看所有局部变量
```

### 2.4 观察点（Watchpoint）

```bash
# 当变量值发生变化时中断
(lldb) watchpoint set variable sum

# 观察内存地址
(lldb) watchpoint set expression -- &sum

# 列出观察点
(lldb) watchpoint list

# 删除观察点
(lldb) watchpoint delete 1
```

### 2.5 LLDB与GDB命令对照

| 功能 | GDB | LLDB |
|------|-----|------|
| 设断点 | `break main` | `b main` |
| 运行 | `run` | `run` |
| 下一步 | `next` / `n` | `next` / `n` |
| 进入函数 | `step` / `s` | `step` / `s` |
| 打印变量 | `print var` | `p var` / `frame variable var` |
| 调用栈 | `backtrace` / `bt` | `bt` |
| 继续 | `continue` / `c` | `continue` / `c` |
| 退出 | `quit` | `quit` / `q` |

## 三、注意事项与常见陷阱

1. **macOS默认调试器**：Xcode和macOS命令行工具自带LLDB，GDB在macOS上配置困难
2. **命令缩写**：LLDB支持命令缩写，`b`、`n`、`s`、`c`等与GDB一致
3. **Tab补全**：LLDB支持Tab键补全命令，比GDB更友好
4. **Python脚本**：LLDB支持Python脚本扩展，可以自定义调试命令
5. **DWARF调试信息**：确保编译时加`-g`选项，LLDB依赖DWARF调试信息
