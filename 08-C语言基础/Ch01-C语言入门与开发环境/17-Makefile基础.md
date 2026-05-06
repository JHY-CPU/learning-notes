# Makefile基础

## 一、Make工具概述

`make` 是一个构建自动化工具，通过读取 `Makefile` 文件来管理和编译项目。它能自动判断哪些文件需要重新编译，避免重复编译未修改的文件。

### 1.1 基本概念

```bash
# 安装make
sudo apt install make        # Ubuntu/Debian
sudo yum install make        # CentOS/RHEL
# macOS自带
# Windows: MinGW或MSYS2
```

## 二、第一个Makefile

### 2.1 简单示例

```makefile
# Makefile
hello: hello.c
	gcc hello.c -o hello

clean:
	rm -f hello
```

```bash
# 使用
make          # 编译hello
make clean    # 清理
```

### 2.2 Makefile规则

```makefile
target: prerequisites
	command
	command
```

```
┌──────────────────────────────────────────┐
│ target: prerequisites                    │
│     command                              │
│                                          │
│ target:     目标文件（要生成的文件）        │
│ prerequisites: 依赖文件（目标文件依赖的）   │
│ command:    生成目标的命令（必须缩进）       │
└──────────────────────────────────────────┘
```

## 三、规则详解

### 3.1 基本规则

```makefile
# 规则：目标: 依赖
# 命令必须以Tab开头（不是空格！）

program: main.o utils.o math.o
	gcc main.o utils.o math.o -o program

main.o: main.c utils.h
	gcc -c main.c -o main.o

utils.o: utils.c utils.h
	gcc -c utils.c -o utils.o

math.o: math.c math.h
	gcc -c math.c -o math.o

clean:
	rm -f *.o program
```

### 3.2 make的执行逻辑

```
make program 的执行过程：

1. 检查 program 是否存在
2. 检查 main.o, utils.o, math.o 是否需要更新
3. 对每个 .o 文件：
   - 检查对应的 .c 文件是否更新
   - 如果更新，重新编译
4. 如果任何 .o 文件更新了，重新链接 program

依赖关系：
program → main.o   → main.c, utils.h
        → utils.o  → utils.c, utils.h
        → math.o   → math.c, math.h
```

### 3.3 时间戳检查

```
make通过比较文件的时间戳判断是否需要重新编译：

如果 target 不存在 → 执行命令
如果 prerequisite 比 target 新 → 执行命令
如果 prerequisite 不存在 → 递归查找prerequisite的规则
```

## 四、变量

### 4.1 变量定义

```makefile
# 简单变量（立即展开）
CC := gcc
CFLAGS := -Wall -Wextra -g

# 递延变量（使用时展开）
SRC = main.c utils.c math.c
OBJ = $(SRC:.c=.o)

# 追加变量
CFLAGS += -std=c11
LDFLAGS += -lm
```

### 4.2 自动变量

```makefile
# $@  当前目标名
# $<  第一个依赖
# $^  所有依赖
# $?  比目标新的依赖
# $*  模式匹配的茎（stem）

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@
# 等价于：
# gcc $(CFLAGS) -c <源文件> -o <目标文件>

program: main.o utils.o
	gcc $^ -o $@
# 等价于：
# gcc main.o utils.o -o program
```

### 4.3 预定义变量

```makefile
# 常用预定义变量
CC = cc          # C编译器
CXX = c++        # C++编译器
AR = ar          # 归档工具
RM = rm -f       # 删除命令
CFLAGS =         # C编译选项
CXXFLAGS =       # C++编译选项
LDFLAGS =        # 链接选项
CPPFLAGS =       # 预处理选项
```

## 五、模式规则

### 5.1 模式规则

```makefile
# 模式规则：%.o 匹配任意 .o 文件
%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

# 使用模式规则后的简化Makefile
CC = gcc
CFLAGS = -Wall -Wextra -g -std=c11
SRCS = main.c utils.c math.c
OBJS = $(SRCS:.c=.o)
TARGET = program

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
```

### 5.2 隐式规则

Make有内置的隐式规则，以下两个Makefile等价：

```makefile
# 显式写法
main.o: main.c
	gcc -c main.c -o main.o

# 隐式规则（make会自动推导）
# 只需要写出依赖关系
main.o: main.c utils.h
```

```bash
# 查看make使用的规则
make -n --debug=b program
make -p | grep "\.c"
```

## 六、函数

### 6.1 常用函数

```makefile
SRCS = main.c utils.c math.c

# patsubst: 模式替换
OBJS = $(patsubst %.c,%.o,$(SRCS))
# 或使用变量替换引用
OBJS = $(SRCS:.c=.o)

# wildcard: 通配符展开
SRCS = $(wildcard *.c)
SRCS = $(wildcard src/*.c)

# notdir: 去掉目录部分
FILE = $(notdir src/main.c)    # 结果: main.c

# dir: 取目录部分
DIR = $(dir src/main.c)        # 结果: src/

# basename: 取文件名（去掉后缀）
NAME = $(basename main.c)      # 结果: main

# addprefix: 添加前缀
INCLUDES = $(addprefix -I, include lib/include)
# 结果: -Iinclude -Ilib/include

# addsuffix: 添加后缀
DEPS = $(addsuffix .d, main utils math)
# 结果: main.d utils.d math.d

# foreach: 循环
DIRS = src lib test
SRCS = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.c))

# if: 条件
DEBUG = 1
CFLAGS = $(if $(DEBUG),-g -DDEBUG,-O2 -DNDEBUG)
```

## 七、条件判断

```makefile
# ifdef / ifndef
ifdef DEBUG
	CFLAGS += -g -DDEBUG
else
	CFLAGS += -O2 -DNDEBUG
endif

# ifeq / ifneq
CC = gcc
ifeq ($(CC),gcc)
	CFLAGS += -Wall
else ifeq ($(CC),clang)
	CFLAGS += -Wall -Wextra
else
	$(error Unknown compiler: $(CC))
endif

# 使用命令行变量
# make DEBUG=1
# make CC=clang
```

## 八、完整Makefile示例

### 8.1 项目结构

```
project/
├── Makefile
├── src/
│   ├── main.c
│   ├── utils.c
│   └── math_ops.c
├── include/
│   ├── utils.h
│   └── math_ops.h
├── lib/
│   └── libhelper.a
└── build/
```

### 8.2 完整Makefile

```makefile
# ==================== 项目配置 ====================
TARGET = myprogram
CC = gcc
AR = ar

# ==================== 目录配置 ====================
SRC_DIR = src
INC_DIR = include
LIB_DIR = lib
BUILD_DIR = build

# ==================== 编译选项 ====================
CFLAGS = -Wall -Wextra -std=c11 -I$(INC_DIR)
LDFLAGS = -L$(LIB_DIR) -lhelper -lm

# Debug/Release配置
ifdef DEBUG
	CFLAGS += -g -Og -DDEBUG
	BUILD_DIR = build/debug
else
	CFLAGS += -O2 -DNDEBUG
	BUILD_DIR = build/release
endif

# ==================== 文件列表 ====================
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS = $(OBJS:.o=.d)

# ==================== 目标规则 ====================
.PHONY: all clean debug release

all: $(BUILD_DIR)/$(TARGET)

debug:
	$(MAKE) all DEBUG=1

release:
	$(MAKE) all

$(BUILD_DIR)/$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	@echo "构建完成: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# ==================== 清理 ====================
clean:
	rm -rf build/

# ==================== 依赖文件 ====================
-include $(DEPS)

# ==================== 其他目标 ====================
run: all
	./$(BUILD_DIR)/$(TARGET)

test: all
	./$(BUILD_DIR)/$(TARGET) --test

install: all
	cp $(BUILD_DIR)/$(TARGET) /usr/local/bin/

.PHONY: help
help:
	@echo "可用目标:"
	@echo "  all     - 构建项目（默认）"
	@echo "  debug   - Debug构建"
	@echo "  release - Release构建"
	@echo "  clean   - 清理构建文件"
	@echo "  run     - 构建并运行"
	@echo "  test    - 运行测试"
	@echo "  help    - 显示此帮助"
```

## 九、依赖自动推导

### 9.1 -MMD -MP 选项

```makefile
# 使用 -MMD 自动生成依赖关系
%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# -MMD 生成 .d 依赖文件
# -MP 为每个头文件生成伪目标，防止删除头文件后出错

# 包含自动生成的依赖
-include $(DEPS)
```

### 9.2 依赖文件内容示例

```
# main.d 文件内容
main.o: main.c include/utils.h include/math_ops.h

include/utils.h:
include/math_ops.h:
```

## 十、常用技巧

### 10.1 .PHONY 声明

```makefile
# .PHONY 声明伪目标，防止与同名文件冲突
.PHONY: all clean test run

clean:
	rm -f *.o $(TARGET)

# 如果目录中恰好有一个叫 clean 的文件，
# 没有.PHONY声明的话 make clean 会认为已经是最新的
```

### 10.2 多行命令

```makefile
define PRINT_INFO
	@echo "============================="
	@echo "目标: $@"
	@echo "编译器: $(CC)"
	@echo "选项: $(CFLAGS)"
	@echo "============================="
endef

$(TARGET): $(OBJS)
	$(PRINT_INFO)
	$(CC) $(CFLAGS) $^ -o $@
```

### 10.3 并行编译

```bash
# 使用多核并行编译
make -j4          # 4个并行任务
make -j$(nproc)   # 使用所有CPU核心
```

## 十一、常见问题

### 11.1 Tab vs 空格

```
Makefile中的命令必须以Tab开头，不能用空格！

错误：
main.o: main.c
    gcc -c main.c -o main.o    ← 这是空格

正确：
main.o: main.c
	gcc -c main.c -o main.o    ← 这是Tab
```

### 11.2 变量展开

```makefile
# 立即展开（:=）
X := hello
Y := $(X) world    # Y = "hello world"

# 递延展开（=）
X = hello
Y = $(X) world     # Y展开时才查找X的值

# 条件赋值（?=）
CC ?= gcc          # 如果CC未定义则赋值

# 追加（+=）
CFLAGS += -Wall    # 追加到已有值
```

## 十二、关键要点

> **重要提示**：
> 1. Makefile规则格式：`target: prerequisites` + Tab命令
> 2. `$@` 是目标名，`$<` 是第一个依赖，`$^` 是所有依赖
> 3. 使用 `%.o: %.c` 模式规则简化重复的编译规则
> 4. 使用 `-MMD` 自动生成头文件依赖
> 5. `.PHONY` 声明伪目标避免与文件名冲突
> 6. 命令前加 `@` 隐藏命令回显
> 7. 使用 `make -j` 并行编译加快速度
> 8. 命令必须以Tab开头，不能用空格
> 9. `wildcard` 和 `patsubst` 函数常用与文件列表处理
> 10. Debug和Release配置使用条件判断区分
