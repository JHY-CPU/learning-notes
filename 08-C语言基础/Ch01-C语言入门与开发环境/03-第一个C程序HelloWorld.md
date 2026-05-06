# 第一个C程序 - Hello World

## 一、Hello World程序

### 1.1 完整代码

```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

### 1.2 逐行解析

#### 第1行：`#include <stdio.h>`

```c
#include <stdio.h>  /* 预处理指令：包含标准输入输出头文件 */
```

- `#include` 是预处理器指令，在编译之前执行
- `stdio.h` 是 **St**andar**d** **I**nput **O**utput **H**eader 的缩写
- `<>` 表示从系统标准路径查找头文件
- 包含了 `printf`、`scanf` 等函数的声明
- 没有这行，编译器不知道 `printf` 是什么

#### 第2行：空行

空行用于提高代码可读性，分隔预处理指令和函数定义。

#### 第3行：`int main(void)`

```c
int main(void)
│    │    │
│    │    └── void 表示不接受参数
│    └─────── main 是程序入口函数名
└──────────── int 表示函数返回整数类型
```

- `main` 函数是每个C程序的**入口点**
- 操作系统通过调用 `main` 函数来启动程序
- `int` 表示 `main` 函数返回一个整数值给操作系统
- `void` 表示 `main` 函数不接受任何参数
- 也可以写成 `int main()`（C语言中含义相同）

#### 第4行：`{`

左花括号表示函数体的开始。

#### 第5行：`printf("Hello, World!\n");`

```c
printf("Hello, World!\n");
│      │              │  │
│      │              │  └── 语句结束符（分号）
│      │              └───── \n 换行符
│      └──────────────────── 要输出的字符串
└─────────────────────────── 标准输出函数名
```

- `printf` 是格式化输出函数，定义在 `stdio.h` 中
- `"Hello, World!"` 是要输出的字符串字面量
- `\n` 是转义字符，表示换行（newline）
- 每条C语句必须以分号 `;` 结尾
- 双引号内的内容称为**字符串常量**

#### 第6行：`return 0;`

```c
return 0;
│       │
│       └── 返回值为0，表示程序成功执行
└────────── return 关键字，从函数返回
```

- `return` 语句结束函数的执行
- 返回值 `0` 表示程序**正常结束**
- 返回非零值通常表示程序**异常结束**
- 这个返回值传递给操作系统，可在shell中用 `$?`（Linux）或 `%ERRORLEVEL%`（Windows）查看

#### 第7行：`}`

右花括号表示函数体的结束。

## 二、编译和运行

### 2.1 编译运行流程

```
源代码(.c)  →  预处理  →  编译  →  汇编  →  链接  →  可执行文件
hello.c        展开宏     汇编代码   目标文件    合并库    hello.exe/a.out
```

### 2.2 在Linux/macOS上

```bash
# 1. 编写源代码
vim hello.c

# 2. 编译（使用gcc）
gcc hello.c -o hello

# 3. 运行
./hello

# 输出：
# Hello, World!
```

### 2.3 在Windows上

```cmd
:: 使用MinGW的gcc
gcc hello.c -o hello.exe

:: 运行
hello.exe

:: 或者使用Visual Studio的cl编译器
cl hello.c
hello.exe
```

### 2.4 分开编译步骤

```bash
# 只预处理（生成 .i 文件）
gcc -E hello.c -o hello.i

# 只编译（生成汇编代码 .s 文件）
gcc -S hello.c -o hello.s

# 只汇编（生成目标文件 .o/.obj）
gcc -c hello.c -o hello.o

# 链接（生成可执行文件）
gcc hello.o -o hello
```

## 三、程序的执行过程

```
┌─────────────────────────────────────────────────────┐
│                     操作系统                          │
│                                                      │
│  1. 用户输入 ./hello 或 hello.exe                     │
│  2. 操作系统加载可执行文件到内存                        │
│  3. 操作系统调用 main() 函数                          │
│         ↓                                            │
│  ┌──────────────────────────────────────┐            │
│  │           程序的执行                  │            │
│  │                                      │            │
│  │  4. printf() 输出 "Hello, World!\n"  │            │
│  │  5. return 0 返回给操作系统          │            │
│  │                                      │            │
│  └──────────────────────────────────────┘            │
│         ↓                                            │
│  6. 操作系统收到返回值0，程序终止                       │
└─────────────────────────────────────────────────────┘
```

## 四、代码变体

### 4.1 多行输出

```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    printf("你好，世界！\n");
    printf("This is my first C program.\n");
    return 0;
}
```

### 4.2 使用格式化输出

```c
#include <stdio.h>

int main(void) {
    char name[] = "小明";
    int age = 20;

    printf("大家好，我叫%s，今年%d岁。\n", name, age);
    printf("我在学习C语言！\n");

    return 0;
}
```

### 4.3 main函数的其他合法形式

```c
/* 形式1：最常用 */
int main(void) {
    return 0;
}

/* 形式2：接受命令行参数 */
int main(int argc, char *argv[]) {
    /* argc: 参数个数 */
    /* argv: 参数数组 */
    printf("程序名: %s\n", argv[0]);
    printf("参数个数: %d\n", argc);
    return 0;
}

/* 形式3：C99允许省略return（隐式返回0）*/
int main(void) {
    printf("Hello\n");
    /* 编译器会自动添加 return 0; （C99及以上）*/
}

/* 形式4：void main（非标准，不推荐）*/
/* 某些编译器支持，但不符合C标准 */
```

## 五、常见初学者问题

### 5.1 忘写分号

```c
/* 错误 */
printf("Hello, World!\n")   /* 缺少分号 */

/* 正确 */
printf("Hello, World!\n");
```

编译错误信息：
```
error: expected ';' before 'return'
```

### 5.2 忘写#include

```c
/* 错误：没有包含头文件 */
int main(void) {
    printf("Hello\n");  /* 警告：隐式声明printf */
    return 0;
}
```

编译警告信息：
```
warning: implicit declaration of function 'printf'
```

### 5.3 中文标点符号

```c
/* 错误：使用了中文分号 */
int main(void) {
    printf("Hello\n")；  /* 中文分号，编译失败 */
    return 0；
}
```

### 5.4 main函数拼写错误

```c
/* 错误 */
int mian(void) {   /* 拼写错误：mian -> main */
    printf("Hello\n");
    return 0;
}
```

链接错误信息：
```
undefined reference to `WinMain'
```

### 5.5 字符串引号不匹配

```c
/* 错误 */
printf("Hello, World!\n');   /* 引号不匹配 */

/* 正确 */
printf("Hello, World!\n");
```

## 六、printf函数详解

### 6.1 常用格式说明符

```c
#include <stdio.h>

int main(void) {
    int integer = 42;
    float floating = 3.14;
    char character = 'A';
    char string[] = "Hello";
    double double_val = 3.141592653589793;

    /* 各种格式说明符 */
    printf("整数:      %d\n", integer);       /* %d 或 %i 整数 */
    printf("浮点数:    %f\n", floating);       /* %f 浮点数 */
    printf("字符:      %c\n", character);     /* %c 字符 */
    printf("字符串:    %s\n", string);         /* %s 字符串 */
    printf("双精度:    %lf\n", double_val);   /* %lf 双精度 */
    printf("八进制:    %o\n", integer);        /* %o 八进制 */
    printf("十六进制:  %x\n", integer);        /* %x 十六进制 */
    printf("指针地址:  %p\n", (void*)&integer); /* %p 指针 */
    printf("百分号:    %%\n");                  /* %% 输出%本身 */

    return 0;
}
```

### 6.2 转义字符

```c
#include <stdio.h>

int main(void) {
    printf("换行符: Hello\\n World\n");       /* \n 换行 */
    printf("制表符: Name\\tAge\n");            /* \t 制表 */
    printf("退格:   ABC\bD\n");                /* \b 退格 */
    printf("回车:   ABC\rD\n");                /* \r 回车 */
    printf("反斜杠: path\\to\\file\n");        /* \\ 反斜杠 */
    printf("引号:   She said \"Hi\"\n");      /* \" 双引号 */
    printf("响铃:   \a");                      /* \a 响铃 */
    return 0;
}
```

## 七、关键要点

> **重要提示**：
> 1. `#include <stdio.h>` 是使用 `printf` 函数的前提
> 2. `main` 函数是每个C程序的入口点，有且只有一个
> 3. 每条C语句以分号 `;` 结尾
> 4. `return 0` 表示程序正常结束
> 5. `printf` 中的 `\n` 是换行符，不是字母n
> 6. 编译命令：`gcc hello.c -o hello`
> 7. 注意区分中文标点和英文标点
> 8. C语言是大小写敏感的：`main` 不等于 `Main`
