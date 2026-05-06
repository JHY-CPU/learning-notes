# 25 - extern 关键字

## 一、extern 基本概念

`extern` 用于声明一个在**其他文件**（或同一文件其他位置）中定义的全局变量或函数，表示"这个东西存在，但定义在别处"。

```c
// 声明（不分配内存）
extern int global_var;

// 定义（分配内存，在另一个文件中）
int global_var = 42;
```

---

## 二、extern 声明变量

### 2.1 跨文件共享变量

```c
// ===== globals.c =====
int shared_count = 0;          // 定义（分配内存）
const char *app_name = "MyApp";

void increment(void) {
    shared_count++;
}

// ===== main.c =====
extern int shared_count;       // 声明（不分配内存）
extern const char *app_name;   // 声明
void increment(void);          // 函数声明（extern 可省略）

int main(void) {
    printf("%s: %d\n", app_name, shared_count);
    increment();
    printf("计数: %d\n", shared_count);
    return 0;
}
```

### 2.2 声明与定义的区别

| | 定义 | 声明 |
|---|------|------|
| `extern` | 可省略（无初始化时） | 必须使用 |
| 内存分配 | 是 | 否 |
| 初始化 | 可以初始化 | 不能初始化 |
| 重复 | 同一作用域只能一次 | 可以多次 |

```c
// 在函数/文件作用域中
int x = 10;            // 定义（分配内存）
extern int y = 20;     // 也是定义（有初始化）
extern int z;          // 声明（没有初始化）
```

---

## 三、extern 声明函数

函数声明默认就是 `extern` 的，通常省略：

```c
// 以下三者等价
void func(int x);
extern void func(int x);

int add(int a, int b);
extern int add(int a, int b);
```

---

## 四、extern 与头文件

### 4.1 推荐模式

```c
// ===== config.h =====
#ifndef CONFIG_H
#define CONFIG_H

// 声明（放在头文件中）
extern int global_timeout;
extern const char *version;
extern int debug_mode;

void set_timeout(int seconds);
const char *get_version(void);

#endif

// ===== config.c =====
#include "config.h"

// 定义（放在源文件中）
int global_timeout = 30;
const char *version = "1.0.0";
int debug_mode = 0;

void set_timeout(int seconds) {
    global_timeout = seconds;
}

const char *get_version(void) {
    return version;
}
```

```c
// ===== main.c =====
#include "config.h"    // 获取声明

int main(void) {
    printf("版本: %s\n", version);
    set_timeout(60);
    return 0;
}
```

---

## 五、extern 的链接属性

### 5.1 外部链接（External Linkage）

```c
// 全局变量和非 static 函数默认具有外部链接
int global_var = 42;          // 外部链接
void public_func(void) {}     // 外部链接

extern int global_var;        // 可以在其他文件中声明
```

### 5.2 内部链接（Internal Linkage）

```c
// static 修饰的变量和函数具有内部链接
static int local_var = 42;        // 内部链接
static void helper(void) {}       // 内部链接

// 其他文件中无法通过 extern 访问
extern int local_var;             // 错误！链接不到
```

---

## 六、extern 与 const

### 6.1 const 的特殊性

```c
// const 全局变量默认是内部链接（C 中）
const int MAX = 100;       // 内部链接（类似 static）
// 其他文件中 extern const int MAX; 会找不到定义！

// 需要显式使用 extern
extern const int MAX = 100;  // 外部链接
```

### 6.2 实际应用

```c
// ===== constants.h =====
extern const int MAX_BUFFER;
extern const double PI;

// ===== constants.c =====
extern const int MAX_BUFFER = 4096;
extern const double PI = 3.14159265358979323846;
```

---

## 七、extern 与数组

```c
// ===== data.c =====
int arr[100] = {0};

// ===== main.c =====
extern int arr[100];      // 完整声明
// 或
extern int arr[];         // 不完整声明（大小可省略）
```

---

## 八、extern 与 struct

```c
// ===== types.h =====
struct Point {
    int x, y;
};

extern struct Point origin;   // 声明

// ===== types.c =====
struct Point origin = {0, 0}; // 定义
```

---

## 九、extern 常见错误

### 9.1 重复定义

```c
// ===== file1.c =====
int counter = 0;

// ===== file2.c =====
int counter = 0;    // 错误！重复定义
// 正确：extern int counter;
```

### 9.2 声明但未定义

```c
// main.c
extern int missing_var;   // 声明了但没有文件定义

int main(void) {
    printf("%d\n", missing_var);  // 链接错误：undefined reference
    return 0;
}
```

### 9.3 类型不匹配

```c
// ===== file1.c =====
double value = 3.14;

// ===== file2.c =====
extern int value;       // 错误！类型不匹配
printf("%d\n", value);  // 未定义行为
```

### 9.4 const 问题

```c
// ===== file1.c =====
const int LIMIT = 100;   // 默认内部链接！

// ===== file2.c =====
extern const int LIMIT;  // 链接错误：找不到定义
// 正确：file1.c 中应写 extern const int LIMIT = 100;
```

---

## 十、extern "C"（C++ 中使用）

在 C++ 中，`extern "C"` 用于告诉编译器使用 C 语言的链接约定：

```c
// 在 C++ 中调用 C 函数
extern "C" {
    #include "my_c_library.h"
}

// 或在头文件中
#ifdef __cplusplus
extern "C" {
#endif

void c_function(int x);
int c_api(void);

#ifdef __cplusplus
}
#endif
```

---

## 十一、最佳实践

1. **头文件放声明，源文件放定义**
2. **尽量少用全局变量**，用函数参数/返回值代替
3. **const 变量跨文件使用时**，记得加 `extern`
4. **不要在头文件中定义变量**（会导致重复定义）
5. **使用头文件保护**防止重复包含

```c
// 推荐的项目结构
// ===== mymodule.h =====
#ifndef MYMODULE_H
#define MYMODULE_H

extern int module_status;
void module_init(void);
void module_cleanup(void);

#endif

// ===== mymodule.c =====
#include "mymodule.h"

int module_status = 0;

void module_init(void) {
    module_status = 1;
}

void module_cleanup(void) {
    module_status = 0;
}
```

---

## 十二、要点总结

1. `extern` 用于声明在其他文件中定义的全局变量和函数
2. 声明不分配内存，定义才分配内存
3. 全局变量和函数默认具有**外部链接**，`static` 改为内部链接
4. `const` 全局变量默认是内部链接，跨文件需要显式 `extern`
5. 头文件中放声明（`extern`），源文件中放定义
6. 注意声明和定义的类型必须一致
7. `extern` 可以在变量定义时省略，但声明时建议加上
8. 在 C++ 中使用 `extern "C"` 实现 C/C++ 混合编程
