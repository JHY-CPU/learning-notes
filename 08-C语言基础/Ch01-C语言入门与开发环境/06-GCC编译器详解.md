# GCC编译器详解

## 一、GCC简介

GCC（GNU Compiler Collection）是GNU项目开发的编译器套件，是最广泛使用的开源编译器。它支持多种编程语言，包括C、C++、Fortran、Ada、Go等。

### 1.1 安装检查

```bash
# 检查是否已安装
gcc --version

# 查看详细信息
gcc -v

# 查看支持的机器架构
gcc -dumpmachine

# 查看默认搜索路径
gcc -print-search-dirs
```

## 二、基本编译命令

### 2.1 最简单的编译

```bash
# 编译并链接，生成 a.out（Linux/macOS）或 a.exe（Windows）
gcc hello.c

# 指定输出文件名
gcc hello.c -o hello

# Windows上
gcc hello.c -o hello.exe
```

### 2.2 分步编译

```bash
# 第1步：预处理（-E）
gcc -E hello.c -o hello.i

# 第2步：编译为汇编代码（-S）
gcc -S hello.i -o hello.s
gcc -S hello.c -o hello.s    # 也可以直接从 .c 生成

# 第3步：汇编为目标文件（-c）
gcc -c hello.s -o hello.o
gcc -c hello.c -o hello.o    # 也可以直接从 .c 生成

# 第4步：链接
gcc hello.o -o hello
```

### 2.3 多文件编译

```bash
# 方法1：直接编译所有源文件
gcc main.c utils.c math_ops.c -o program

# 方法2：先分别编译，再链接
gcc -c main.c -o main.o
gcc -c utils.c -o utils.o
gcc -c math_ops.c -o math_ops.o
gcc main.o utils.o math_ops.o -o program

# 方法3：使用通配符（不推荐用于大型项目）
gcc *.c -o program
```

## 三、常用编译选项

### 3.1 警告选项

```bash
# 开启基本警告
gcc -Wall hello.c -o hello

# -Wall 包含的警告：
# -Wcomment        注释嵌套警告
# -Wformat         printf/scanf格式化警告
# -Wmain           main函数问题警告
# -Wmissing-braces 数组初始化括号警告
# -Wparentheses    括号优先级警告
# -Wreturn-type    返回类型警告
# -Wsequence-point 序列点警告
# -Wswitch         switch语句警告
# -Wuninitialized  未初始化变量警告
# ... 等等

# 开启额外警告
gcc -Wall -Wextra hello.c -o hello

# -Wextra 包含的额外警告：
# -Wsign-compare        有符号/无符号比较
# -Wtype-limits         类型范围检查
# -Wempty-body          空循环体
# -Wignored-qualifiers  忽略的类型限定符

# 将警告视为错误
gcc -Wall -Werror hello.c -o hello

# 特定警告
gcc -Wunused-variable hello.c -o hello
gcc -Wshadow hello.c -o hello           # 变量遮蔽警告

# 禁用特定警告
gcc -Wall -Wno-unused-variable hello.c -o hello
```

**推荐的警告组合**：
```bash
# 开发时推荐
gcc -Wall -Wextra -Wpedantic hello.c -o hello

# 严格模式
gcc -Wall -Wextra -Werror -Wpedantic hello.c -o hello
```

### 3.2 优化选项

```bash
# 无优化（默认，-O0）
gcc -O0 hello.c -o hello

# 基本优化（-O1）
gcc -O1 hello.c -o hello

# 更多优化（-O2，推荐用于发布）
gcc -O2 hello.c -o hello

# 最大优化（-O3）
gcc -O3 hello.c -o hello

# 优化代码大小（-Os）
gcc -Os hello.c -o hello

# 优化调试体验（-Og，推荐用于调试）
gcc -Og hello.c -o hello
```

**优化级别说明**：

| 级别 | 说明 | 编译速度 | 执行速度 | 调试难度 |
|------|------|---------|---------|---------|
| -O0 | 无优化 | 最快 | 最慢 | 最容易 |
| -O1 | 基本优化 | 快 | 较快 | 较容易 |
| -O2 | 推荐优化 | 中等 | 快 | 较难 |
| -O3 | 最大优化 | 慢 | 最快 | 最难 |
| -Os | 大小优化 | 中等 | 快 | 较难 |
| -Og | 调试优化 | 快 | 较快 | 容易 |

### 3.3 调试选项

```bash
# 生成调试信息（-g）
gcc -g hello.c -o hello

# 调试级别
gcc -g1 hello.c -o hello   # 最少调试信息
gcc -g2 hello.c -o hello   # 默认级别
gcc -g3 hello.c -o hello   # 最多调试信息（含宏）

# 使用DWARF调试格式
gcc -gdwarf-4 hello.c -o hello

# 推荐：调试编译选项
gcc -Wall -Wextra -g -Og hello.c -o hello
```

### 3.4 标准指定

```bash
# 指定C标准
gcc -std=c89 hello.c -o hello    # C89/C90标准
gcc -std=c99 hello.c -o hello    # C99标准
gcc -std=c11 hello.c -o hello    # C11标准
gcc -std=c17 hello.c -o hello    # C17标准
gcc -std=c2x hello.c -o hello    # C23草案（实验性）

# 严格遵循标准
gcc -std=c11 -pedantic hello.c -o hello

# GNU扩展（默认）
gcc -std=gnu11 hello.c -o hello  # C11 + GNU扩展
```

### 3.5 预处理定义

```bash
# 定义宏
gcc -DDEBUG hello.c -o hello
gcc -DVERSION=2 hello.c -o hello
gcc -DNAME=\"hello\" hello.c -o hello

# 取消定义
gcc -UDEBUG hello.c -o hello

# 等价于代码中的：
#define DEBUG
#define VERSION 2
#define NAME "hello"
```

## 四、头文件和库选项

### 4.1 头文件搜索路径

```bash
# 指定头文件搜索路径
gcc -I/usr/local/include hello.c -o hello
gcc -I./include -I../common/include hello.c -o hello

# 多个路径
gcc -Ipath1 -Ipath2 -Ipath3 hello.c -o hello

# 系统头文件路径（抑制警告）
gcc -isystem /usr/local/include hello.c -o hello
```

### 4.2 库文件搜索路径

```bash
# 指定库文件搜索路径
gcc -L/usr/local/lib hello.c -o hello
gcc -L./lib -L../common/lib hello.c -o hello

# 链接库文件
gcc hello.c -lm -o hello           # 链接数学库 libm
gcc hello.c -lpthread -o hello     # 链接线程库
gcc hello.c -lssl -lcrypto -o hello # 链接OpenSSL

# 指定静态链接库
gcc hello.c -l:libmylib.a -o hello
```

### 4.3 链接选项

```bash
# 静态链接
gcc -static hello.c -o hello

# 指定共享库
gcc hello.c -Wl,-rpath,/usr/local/lib -o hello

# 指定动态链接器
gcc -Wl,-dynamic-linker,/lib64/ld-linux-x86-64.so.2 hello.c -o hello

# 导出所有符号
gcc -Wl,--export-all-symbols hello.c -o hello
```

## 五、输出文件选项

```bash
# 指定输出文件名
gcc hello.c -o hello

# 只生成目标文件
gcc -c hello.c -o hello.o

# 生成汇编代码
gcc -S hello.c -o hello.s

# 生成预处理后的文件
gcc -E hello.c -o hello.i

# 生成依赖关系文件
gcc -M hello.c                    # 包含系统头文件
gcc -MM hello.c                   # 只包含用户头文件
gcc -MD -c hello.c                # 生成 .d 依赖文件
```

## 六、高级选项

### 6.1 代码生成选项

```bash
# 位置无关代码（用于共享库）
gcc -fPIC hello.c -c -o hello.o

# 生成32位代码
gcc -m32 hello.c -o hello

# 生成64位代码
gcc -m64 hello.c -o hello

# 指定CPU架构
gcc -march=native hello.c -o hello      # 当前CPU
gcc -march=x86-64 hello.c -o hello      # 通用x86-64
gcc -march=armv8-a hello.c -o hello     # ARM v8
```

### 6.2 链接时优化（LTO）

```bash
# 启用链接时优化
gcc -flto -O2 hello.c -o hello

# 指定并行LTO
gcc -flto -O2 -flto=4 hello.c -o hello

# 分步使用LTO
gcc -flto -O2 -c hello.c -o hello.o    # 编译
gcc -flto -O2 hello.o -o hello          # 链接
```

### 6.3 安全相关选项

```bash
# 栈保护
gcc -fstack-protector hello.c -o hello         # 基本栈保护
gcc -fstack-protector-all hello.c -o hello     # 所有函数栈保护
gcc -fstack-protector-strong hello.c -o hello  # 强栈保护

# 堆保护
gcc -fstack-check hello.c -o hello

# 地址空间布局随机化（ASLR）
gcc -fPIE -pie hello.c -o hello

# 控制流保护
gcc -fcf-protection=full hello.c -o hello

# 不可执行栈
gcc -Wl,-z,noexecstack hello.c -o hello
```

### 6.4 剖析和覆盖率

```bash
# 生成剖析信息（用于 gprof）
gcc -pg hello.c -o hello

# 生成代码覆盖率信息
gcc -fprofile-arcs -ftest-coverage hello.c -o hello
# 运行后生成 .gcda 和 .gcno 文件
# 使用 gcov 分析：
gcov hello.c
```

## 七、实用编译配置

### 7.1 开发配置

```bash
# 日常开发推荐
gcc -Wall -Wextra -Wpedantic -g -Og -std=c11 hello.c -o hello

# 详细开发配置
gcc \
    -Wall -Wextra -Wpedantic \
    -Wshadow -Wformat=2 -Wconversion \
    -g -Og \
    -std=c11 \
    -DDEBUG \
    hello.c -o hello
```

### 7.2 发布配置

```bash
# 发布版本推荐
gcc -Wall -Wextra -O2 -std=c11 -DNDEBUG hello.c -o hello

# 最大性能配置
gcc \
    -Wall -Wextra \
    -O3 -march=native -flto \
    -std=c11 -DNDEBUG \
    -fstack-protector-strong \
    -fPIE -pie \
    hello.c -o hello
```

### 7.3 Makefile中的使用

```makefile
# 通用Makefile片段
CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -std=c11
CFLAGS_DEBUG = -g -Og -DDEBUG
CFLAGS_RELEASE = -O2 -DNDEBUG

LDFLAGS = -lm
TARGET = myprogram
SRCS = main.c utils.c math.c
OBJS = $(SRCS:.c=.o)

# Debug构建
debug: CFLAGS += $(CFLAGS_DEBUG)
debug: $(TARGET)

# Release构建
release: CFLAGS += $(CFLAGS_RELEASE)
release: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
```

## 八、常用GCC工具

```bash
# nm - 查看符号表
nm hello.o
nm -C hello                     # demangle C++符号

# objdump - 反汇编
objdump -d hello                # 反汇编代码段
objdump -h hello                # 显示段头
objdump -t hello                # 显示符号表
objdump -S hello                # 混合源码和汇编（需要-g）

# readelf - 查看ELF文件
readelf -h hello                # ELF头信息
readelf -S hello                # 段信息
readelf -s hello                # 符号表

# strings - 提取字符串
strings hello                   # 提取可打印字符串

# strip - 去除符号
strip hello                     # 去除调试符号，减小文件大小

# addr2line - 地址转行号
addr2line -e hello 0x401234     # 将地址转换为源码行号
```

## 九、常见问题

### 9.1 编译命令找不到

```bash
# 检查安装
which gcc
type gcc

# Ubuntu/Debian安装
sudo apt install build-essential

# CentOS/RHEL安装
sudo yum install gcc make
```

### 9.2 头文件找不到

```bash
# 查看默认头文件搜索路径
cpp -v /dev/null -o /dev/null

# 指定头文件路径
gcc -I/path/to/headers hello.c -o hello
```

### 9.3 链接库找不到

```bash
# 查看库搜索路径
ldconfig -p | grep libm

# 指定库路径
gcc -L/path/to/libs -lmylib hello.c -o hello

# 查看运行时库依赖
ldd hello
```

## 十、关键要点

> **重要提示**：
> 1. `-Wall -Wextra` 是必须开启的警告选项
> 2. `-g` 生成调试信息，`-Og` 适合调试的优化级别
> 3. `-std=c11` 指定C标准版本
> 4. `-I` 指定头文件路径，`-L` 指定库文件路径
> 5. `-l` 链接库文件，`-static` 静态链接
> 6. `-O2` 是推荐的优化级别，适合发布
> 7. `-fPIC` 用于生成位置无关代码，创建共享库必需
> 8. 开发阶段用 `-g -Og`，发布阶段用 `-O2 -DNDEBUG`
