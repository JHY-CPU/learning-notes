# Makefile基础

## 一、概念说明

Makefile是`make`工具的配置文件，定义了源文件之间的**依赖关系**和**构建规则**。当源文件发生变化时，`make`只会重新编译受影响的文件，大幅提高构建效率。

## 二、具体用法

### 2.1 基本语法

```makefile
# Makefile格式：
# 目标: 依赖
# 	命令（必须以Tab缩进）

# 最简单的Makefile
hello: main.cpp
	g++ -std=c++17 -o hello main.cpp

clean:
	rm -f hello
```

```bash
# 使用make
make         # 构建默认目标（第一个目标）
make hello   # 构建指定目标
make clean   # 执行清理
```

### 2.2 多文件项目的Makefile

```makefile
# 变量定义
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
TARGET = app
SRCS = main.cpp utils.cpp math_ops.cpp
OBJS = $(SRCS:.cpp=.o)

# 默认目标
all: $(TARGET)

# 链接：将目标文件链接为可执行文件
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# 模式规则：.cpp -> .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJS) $(TARGET)

# 声明伪目标（防止与同名文件冲突）
.PHONY: all clean
```

输出示例：
```
$ make
g++ -std=c++17 -Wall -Wextra -g -c main.cpp -o main.o
g++ -std=c++17 -Wall -Wextra -g -c utils.cpp -o utils.o
g++ -std=c++17 -Wall -Wextra -g -c math_ops.cpp -o math_ops.o
g++ -std=c++17 -Wall -Wextra -g -o app main.o utils.o math_ops.o
```

### 2.3 自动变量

```makefile
# 常用自动变量
# $@  目标文件名
# $<  第一个依赖文件名
# $^  所有依赖文件名
# $?  比目标新的依赖文件名
# $*  模式规则中的词干

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
# $< 被替换为当前的.cpp文件名
# $@ 被替换为当前的.o目标文件名
```

### 2.4 依赖头文件的Makefile

```makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
TARGET = app

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# 自动依赖生成
DEPS = $(OBJS:.o=.d)
-include $(DEPS)

%.d: %.cpp
	$(CXX) -MM $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

.PHONY: all clean
```

### 2.5 函数和条件

```makefile
# wildcard函数：获取匹配的文件列表
SRC_FILES = $(wildcard src/*.cpp)

# patsubst函数：模式替换
OBJ_FILES = $(patsubst src/%.cpp, obj/%.o, $(SRC_FILES))

# 条件判断
DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CXXFLAGS += -g -DDEBUG
else
    CXXFLAGS += -O2 -DNDEBUG
endif

obj/%.o: src/%.cpp
	@mkdir -p obj    # @前缀不显示命令
	$(CXX) $(CXXFLAGS) -c $< -o $@
```

## 三、注意事项与常见陷阱

1. **命令必须用Tab缩进**：不能用空格，这是Makefile最常见的错误
2. **.PHONY声明伪目标**：`all`、`clean`等不是实际文件的目标应声明为`.PHONY`
3. **依赖关系要完整**：缺少头文件依赖会导致修改头文件后不重新编译
4. **变量展开时机**：`=`是递归展开（使用时展开），`:=`是简单展开（定义时展开）
5. **并行构建**：`make -j4`使用4个线程并行构建，显著提高大项目编译速度
