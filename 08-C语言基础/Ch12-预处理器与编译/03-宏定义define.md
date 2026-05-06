# 03 - 宏定义 define

## define 指令基础

`#define` 是预处理器中最强大的指令之一，用于定义宏（Macro）。宏的本质是**文本替换**——预处理器会在编译之前将源代码中所有出现的宏名替换为其定义的文本。

## 对象宏（Object-like Macro）

对象宏是最简单的宏形式，用于定义常量或简短的文本替换。

### 基本语法

```c
#define 标识符 替换文本
#define 标识符  // 定义空宏（仅用于条件编译）
```

### 示例

```c
// 常量定义
#define PI 3.14159265358979
#define MAX_BUFFER 1024
#define APP_NAME "MyApplication"
#define NEWLINE '\n'

// 使用
double area = PI * radius * radius;
char buffer[MAX_BUFFER];
printf("Welcome to %s\n", APP_NAME);
```

### 空宏（值为空）

```c
#define DEBUG          // 定义 DEBUG，值为空
#define _GNU_SOURCE    // 启用 GNU 扩展

#ifdef DEBUG
    printf("调试模式\n");
#endif
```

### 多行宏定义

使用反斜杠 `\` 将宏定义延续到下一行：

```c
#define ERROR_MESSAGES \
    "错误1: 文件未找到\n" \
    "错误2: 权限不足\n" \
    "错误3: 内存不足\n"

printf(ERROR_MESSAGES);
```

## 函数宏（Function-like Macro）

函数宏看起来像函数调用，但本质仍是文本替换。

### 基本语法

```c
#define 标识符(参数列表) 替换文本
```

### 示例

```c
// 简单函数宏
#define SQUARE(x) ((x) * (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// 使用
int sq = SQUARE(5);       // 展开为 ((5) * (5)) = 25
int bigger = MAX(3, 7);   // 展开为 ((3) > (7) ? (3) : (7)) = 7
```

### 注意事项：宏名与括号之间不能有空格

```c
// 正确：函数宏
#define SQUARE(x) ((x) * (x))

// 错误：这会被当作对象宏！
#define SQUARE (x) ((x) * (x))
// SQUARE 被定义为 "(x) ((x) * (x))"，不是函数宏
```

## 宏展开的规则

### 展开过程

1. 预处理器扫描源代码中的宏调用
2. 将宏参数替换到宏定义的替换文本中
3. 对替换结果**再次扫描**，展开其中的其他宏（递归展开）

### 展开示例

```c
#define A 10
#define B (A + 5)
#define C (B * 2)

int x = C;
// 展开过程：
// 第1步: C → (B * 2)
// 第2步: B → (A + 5)，得到 ((A + 5) * 2)
// 第3步: A → 10，得到 ((10 + 5) * 2)
// 最终: int x = ((10 + 5) * 2);
```

### 防止无限递归

宏不会无限递归展开：

```c
#define A B
#define B A

int x = A;  // 展开为 B，然后 A...但不会无限循环
// 预处理器会检测到已经展开过的宏，避免重复展开
```

## 宏的文本替换本质

理解宏的核心在于理解它是**纯文本替换**，不涉及任何类型检查或语法分析。

### 示例说明

```c
#define ADD(a, b) a + b

int result = ADD(2, 3) * 4;
// 展开为: int result = 2 + 3 * 4;
// 由于运算优先级，结果是 2 + 12 = 14，不是 (2+3)*4 = 20！
```

### 正确的写法

```c
// 始终用括号包裹整个宏和每个参数
#define ADD(a, b) ((a) + (b))

int result = ADD(2, 3) * 4;
// 展开为: int result = ((2) + (3)) * 4;
// 结果是 5 * 4 = 20 ✓
```

## 宏与类型无关

宏不关心数据类型，这既是优点也是缺点：

```c
#define SQUARE(x) ((x) * (x))

int a = SQUARE(5);        // int: 25
double b = SQUARE(3.14);  // double: 9.8596
float c = SQUARE(2.5f);   // float: 6.25

// 但也有隐患：
int *p = &a;
// SQUARE(*p);  // 展开为 ((*p) * (*p))，可能有副作用
```

## 取消宏定义

使用 `#undef` 取消之前定义的宏：

```c
#define MAX 100
int arr[MAX];  // OK: arr[100]

#undef MAX
// 此后 MAX 未定义
#define MAX 200
int arr2[MAX];  // OK: arr2[200]
```

### 使用场景

```c
// 在特定范围内使用宏
void process(void) {
    #define TEMP_BUFFER_SIZE 512
    char buffer[TEMP_BUFFER_SIZE];
    // ... 使用 buffer
    #undef TEMP_BUFFER_SIZE  // 用完立即取消定义
}
```

## 宏定义的作用域

宏的作用域从定义处开始，到文件末尾或 `#undef` 处结束：

```c
// file1.c
#define VALUE 10   // 在 file1.c 中有效

// file2.c
// VALUE 在这里不可见（除非在头文件中定义）
```

如果宏定义在头文件中，则包含该头文件的所有源文件都拥有该宏。

## 检查宏是否定义

```c
// 方法1: #ifdef
#ifdef MAX_SIZE
    printf("MAX_SIZE is defined as %d\n", MAX_SIZE);
#endif

// 方法2: #if defined()
#if defined(MAX_SIZE) && MAX_SIZE > 100
    printf("MAX_SIZE is greater than 100\n");
#endif

// 方法3: 预定义检查宏
#if defined(__GNUC__)
    printf("Using GCC compiler\n");
#endif
```

## 常见用法模式

### 1. 定义常量

```c
#define PI          3.14159265358979
#define E           2.71828182845905
#define MAX_PATH    260
#define EOF_VALUE   (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
```

### 2. 定义简短的代码片段

```c
#define SWAP(a, b) do { \
    typeof(a) _tmp = (a); \
    (a) = (b); \
    (b) = _tmp; \
} while(0)
```

### 3. 条件编译标志

```c
#define ENABLE_LOGGING 1
#define PLATFORM_LINUX 1
#define VERSION_MAJOR  2
#define VERSION_MINOR  1
```

### 4. 定义关键字替代

```c
// 不推荐，但某些嵌入式系统使用
#define BEGIN {
#define END   }
#define AND   &&
#define OR    ||

if (x > 0 AND y > 0) BEGIN
    printf("both positive\n");
END
```

## 重要注意事项

> **关键点总结**：
> 1. 宏是**纯文本替换**，不是函数调用
> 2. 函数宏的宏名和左括号之间**不能有空格**
> 3. 始终用**括号**包裹宏定义中的参数和整个表达式
> 4. 宏没有类型检查，参数可能被多次求值
> 5. 使用 `#undef` 取消不再需要的宏定义
> 6. 对象宏和函数宏的区别在于定义时宏名后是否有紧跟的括号
