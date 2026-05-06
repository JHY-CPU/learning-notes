# 24 - static 关键字

## 一、static 概述

`static` 是 C 语言中用途最广泛的关键字之一，根据使用位置的不同，有三种不同的含义：

| 使用位置 | 作用 |
|----------|------|
| 修饰局部变量 | 改变生命周期（自动 → 静态） |
| 修饰全局变量 | 改变链接属性（外部 → 内部） |
| 修饰函数 | 改变链接属性（外部 → 内部） |

---

## 二、static 修饰局部变量

### 2.1 改变生命周期

普通局部变量在函数退出时销毁，`static` 局部变量在**程序运行期间一直存在**：

```c
#include <stdio.h>

void counter(void) {
    static int count = 0;   // 只初始化一次
    count++;
    printf("count = %d\n", count);
}

int main(void) {
    counter();  // count = 1
    counter();  // count = 2
    counter();  // count = 3
    return 0;
}
```

### 2.2 对比普通局部变量

| 特性 | 普通局部变量 | static 局部变量 |
|------|-------------|-----------------|
| 存储位置 | 栈 | 数据段 |
| 生命周期 | 函数调用期间 | 程序运行全程 |
| 初始化 | 每次进入函数 | 仅一次 |
| 默认值 | 垃圾值 | 0 |
| 可见性 | 函数内部 | 函数内部（作用域不变） |

### 2.3 实际应用

```c
// 生成唯一 ID
int generate_id(void) {
    static int id = 0;
    return ++id;
}

// 计时
#include <time.h>
void start_timer(void) {
    static clock_t start = 0;
    start = clock();
}

double get_elapsed(void) {
    static clock_t start = 0;
    if (start == 0) {
        start = clock();
        return 0.0;
    }
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}
```

---

## 三、static 修饰全局变量

### 3.1 改变链接属性

全局变量默认具有**外部链接（external linkage）**，其他文件可以通过 `extern` 访问。加上 `static` 后变为**内部链接（internal linkage）**，只能在定义它的文件中访问：

```c
// ===== file1.c =====
static int file_count = 0;    // 只在 file1.c 中可见

void increment(void) {
    file_count++;
}

// ===== file2.c =====
extern int file_count;         // 错误！无法访问 file1.c 中的 static 变量
```

### 3.2 文件作用域的"私有化"

```c
// ===== config.c =====
static int debug_level = 0;       // 只有本文件可以访问
static const char *version = "1.0";

// 提供访问接口
int get_debug_level(void) { return debug_level; }
void set_debug_level(int level) { debug_level = level; }
const char *get_version(void) { return version; }
```

### 3.3 避免命名冲突

```c
// ===== module_a.c =====
static int counter = 0;    // module_a 的 counter
void func_a(void) { counter++; }

// ===== module_b.c =====
static int counter = 0;    // module_b 的 counter（不冲突）
void func_b(void) { counter++; }
```

---

## 四、static 修饰函数

`static` 修饰函数使函数变为**内部链接**，只能在定义它的文件中调用：

```c
// ===== utils.c =====

static int helper(int x) {      // 辅助函数，不对外暴露
    return x * x + 1;
}

static void log_error(const char *msg) {   // 仅本文件使用
    fprintf(stderr, "[ERROR] %s\n", msg);
}

int public_api(int x) {          // 公开接口
    if (x < 0) {
        log_error("负数输入");
        return -1;
    }
    return helper(x);
}
```

### 4.1 static 函数的优点

- **封装性**：隐藏实现细节，只暴露必要的接口
- **避免冲突**：不同文件可以有同名 static 函数
- **编译优化**：编译器可以更好地优化（知道调用范围）

---

## 五、static 的存储位置

```
内存布局：
┌──────────────────┐ 高地址
│     栈（Stack）    │ ← 普通局部变量
├──────────────────┤
│        ↓         │
├──────────────────┤
│     堆（Heap）    │ ← malloc 分配
├──────────────────┤
│        ↑         │
├──────────────────┤
│  BSS 段          │ ← 未初始化的全局/static 变量（自动为 0）
├──────────────────┤
│  数据段（Data）   │ ← 初始化的全局/static 变量
├──────────────────┤
│  代码段（Text）   │ ← 程序代码
└──────────────────┘ 低地址
```

- 普通局部变量：栈
- static 局部变量：数据段或 BSS 段
- static 全局变量：数据段或 BSS 段
- static 函数：代码段（但链接属性受限）

---

## 六、static 的初始化

```c
void func(void) {
    static int x;          // BSS 段，自动初始化为 0
    static int y = 42;     // 数据段，初始化为 42

    // static 变量必须用常量表达式初始化
    int temp = 10;
    // static int z = temp;  // 错误！不是常量表达式
}
```

---

## 七、常见模式

### 7.1 单例模式

```c
// 用 static 实现简单的"单例"
Data *get_instance(void) {
    static Data instance;       // 只创建一次
    static int initialized = 0;

    if (!initialized) {
        init_data(&instance);
        initialized = 1;
    }
    return &instance;
}
```

### 7.2 模块私有状态

```c
// ===== buffer.c =====
static char buffer[1024];
static int buffer_pos = 0;

void buffer_write(const char *data, int len) {
    if (buffer_pos + len <= 1024) {
        memcpy(buffer + buffer_pos, data, len);
        buffer_pos += len;
    }
}

int buffer_read(char *dest, int max_len) {
    int len = (buffer_pos < max_len) ? buffer_pos : max_len;
    memcpy(dest, buffer, len);
    return len;
}
```

---

## 八、要点总结

1. `static` 修饰**局部变量**：改变生命周期，函数退出后仍保留值
2. `static` 修饰**全局变量**：改变链接属性为内部，仅本文件可见
3. `static` 修饰**函数**：改变链接属性为内部，仅本文件可调用
4. static 变量存储在数据段/BSS 段，未初始化时自动为 0
5. static 变量只初始化一次（在程序启动时）
6. 使用 static 可以实现模块的**封装和信息隐藏**
7. static 避免了不同文件之间的命名冲突
8. 合理使用 static 可以提高代码的模块化和可维护性
