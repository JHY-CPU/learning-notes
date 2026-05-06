# 02 - include 指令

## include 指令基础

`#include` 是最常用的预处理指令，用于将指定文件的内容插入到当前文件中。这使得我们可以将声明、宏定义等公共内容放在头文件中，供多个源文件共享。

## 两种包含语法

### 尖括号形式：`#include <filename>`

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
```

- 用于包含**系统头文件**或**标准库头文件**
- 搜索路径：编译器的**标准包含目录**（系统目录）
- 不搜索当前源文件所在目录

### 双引号形式：`#include "filename"`

```c
#include "myheader.h"
#include "utils/utils.h"
```

- 用于包含**用户自定义头文件**
- 搜索路径：首先搜索**当前源文件所在目录**，如果找不到，再按尖括号的方式搜索标准目录
- 可以包含完整或相对路径

## 搜索路径详解

### GCC 搜索顺序

```
1. 对于 #include "file.h":
   (1) 当前源文件所在目录
   (2) -I 选项指定的目录（按命令行顺序）
   (3) 环境变量 CPATH 指定的目录
   (4) 系统默认包含目录

2. 对于 #include <file.h>:
   (1) -I 选项指定的目录
   (2) 环境变量 C_INCLUDE_PATH / CPLUS_INCLUDE_PATH
   (3) 系统默认包含目录（如 /usr/include）
```

### 指定额外搜索路径

```bash
# 使用 -I 选项添加头文件搜索路径
gcc -I./include -I/usr/local/include main.c -o main

# 多个路径用多个 -I
gcc -I./include -I./third_party/include -I/usr/local/include main.c
```

## include 的展开机制

`#include` 指令的效果是**完全复制**被包含文件的内容。预处理器会递归地处理嵌套的 `#include`。

示例：

```c
// myheader.h
#ifndef MYHEADER_H
#define MYHEADER_H

#define MAX_SIZE 100

typedef struct {
    int x;
    int y;
} Point;

#endif
```

```c
// main.c
#include <stdio.h>
#include "myheader.h"

int main(void) {
    Point p = {10, 20};
    printf("Point: (%d, %d)\n", p.x, p.y);
    printf("Max size: %d\n", MAX_SIZE);
    return 0;
}
```

预处理后，`main.c` 相当于：

```c
// stdio.h 的全部内容被粘贴在这里...
// （数以千计的行）

// myheader.h 的全部内容被粘贴在这里：
// #ifndef MYHEADER_H（此时 MYHEADER_H 未定义，条件为真）
// #define MYHEADER_H
// #define MAX_SIZE 100
// typedef struct { int x; int y; } Point;
// #endif

int main(void) {
    Point p = {10, 20};
    printf("Point: (%d, %d)\n", p.x, p.y);
    printf("Max size: %d\n", MAX_SIZE);
    return 0;
}
```

## 嵌套包含与重复包含

### 问题：重复包含

```c
// types.h
typedef struct { int id; char name[50]; } Student;

// file1.c
#include "types.h"
// ... 使用 Student 类型

// file2.c
#include "types.h"
// ... 使用 Student 类型

// main.c
#include "file1.c"  // 不推荐直接 include .c 文件
#include "file2.c"  // 这里会导致 types.h 被包含两次
```

### 解决方案：头文件守卫

```c
// types.h
#ifndef TYPES_H
#define TYPES_H

typedef struct {
    int id;
    char name[50];
} Student;

#endif /* TYPES_H */
```

## include 的最佳实践

### 1. 包含顺序

推荐的包含顺序（Google C++ Style Guide 推荐）：

```c
// 1. 对应的头文件（如果当前是 .c 文件）
#include "mymodule.h"

// 2. C 系统头文件
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 3. 其他库的头文件
#include <sqlite3.h>

// 4. 本项目内的其他头文件
#include "utils/logger.h"
#include "data/database.h"
```

### 2. 最小包含原则

只包含你实际使用的头文件。如果只需要声明（如函数指针类型），考虑使用前向声明。

```c
// 不好：包含了不需要的头文件
#include <stdio.h>   // 只用了 strlen？
#include <string.h>  // 实际只需要这个

// 更好：只包含需要的
#include <string.h>
```

### 3. 自包含头文件

每个头文件应该可以独立编译，即包含它所需要的一切依赖。

```c
// mymodule.h — 自包含
#ifndef MYMODULE_H
#define MYMODULE_H

#include <stddef.h>   // 需要 size_t
#include "config.h"   // 需要 CONFIG_MAX

#define MODULE_VERSION "1.0"
size_t process(const char *input, size_t len);

#endif
```

```c
// 测试自包含性：单独预处理
// gcc -E mymodule.h
```

### 4. 相对路径使用

```c
// 项目结构：
// project/
//   include/
//     utils.h
//     data/
//       db.h
//   src/
//     main.c

// main.c 中：
#include "../include/utils.h"      // 相对路径
#include "../include/data/db.h"    // 嵌套目录

// 更推荐使用 -I 选项：
// gcc -I./include src/main.c
// 然后直接用：
#include "utils.h"
#include "data/db.h"
```

## 常见错误

### 错误1：循环包含

```c
// a.h
#ifndef A_H
#define A_H
#include "b.h"  // a.h 包含 b.h
typedef struct { B *ptr; } A;
#endif

// b.h
#ifndef B_H
#define B_H
#include "a.h"  // b.h 包含 a.h → 循环包含！
typedef struct { A *ptr; } B;
#endif
```

**解决方案**：使用前向声明

```c
// a.h
#ifndef A_H
#define A_H
// 不包含 b.h，而是前向声明
struct B;  // 前向声明
typedef struct { struct B *ptr; } A;
#endif

// b.h
#ifndef B_H
#define B_H
struct A;  // 前向声明
typedef struct { struct A *ptr; } B;
#endif
```

### 错误2：缺少头文件守卫导致重定义

```c
// config.h（没有头文件守卫）
#define BUFFER_SIZE 1024

// main.c
#include "config.h"
#include "utils.h"  // utils.h 也包含了 config.h

// 编译错误：BUFFER_SIZE 重定义
```

### 错误3：include .c 文件

```c
// 不要这样做！
#include "module1.c"
#include "module2.c"

// 正确做法：分别编译，最后链接
// gcc -c module1.c -o module1.o
// gcc -c module2.c -o module2.o
// gcc main.c module1.o module2.o -o program
```

## 预处理器查看工具

```bash
# 查看头文件搜索路径
gcc -v -E -x c /dev/null 2>&1 | grep -A 20 "search starts here"

# 查看特定文件的预处理结果
gcc -E main.c | tail -50

# 查看依赖关系
gcc -M main.c          # 包含系统头文件
gcc -MM main.c         # 只包含用户头文件
gcc -MD main.c         # 生成 .d 依赖文件
```

## 重要注意事项

> **关键点总结**：
> 1. `<>` 和 `""` 的搜索路径不同，选择合适的语法
> 2. 每个头文件必须使用头文件守卫防止重复包含
> 3. 头文件应该自包含——包含它需要的所有依赖
> 4. 避免循环包含，使用前向声明解决
> 5. 使用 `-I` 选项管理项目头文件路径
> 6. 不要 `#include .c` 文件
