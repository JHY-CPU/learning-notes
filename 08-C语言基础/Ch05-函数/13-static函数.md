# 13 - static 函数

## 一、static 修饰函数的作用

在 C 语言中，`static` 关键字用于修饰函数时，表示该函数具有**内部链接（Internal Linkage）**，即该函数只能在**定义它的源文件**中被调用，其他源文件无法访问。

```c
// file1.c
static void helper(void) {
    printf("我是 file1.c 的私有函数\n");
}

void public_func(void) {
    helper();         // 可以调用——在同一个文件中
}
```

```c
// file2.c
// extern void helper(void);  // 链接错误！helper 是 static 的
// helper();                  // 无法调用 file1.c 中的 static 函数
```

## 二、内部链接 vs 外部链接

### 2.1 默认情况：外部链接

```c
// utils.c —— 普通函数，默认具有外部链接
void public_function(void) {
    // 任何文件都可以通过 extern 声明来调用
}
```

```c
// main.c
extern void public_function(void);  // 声明外部函数

int main() {
    public_function();  // 可以调用
    return 0;
}
```

### 2.2 static：内部链接

```c
// utils.c
static void private_helper(void) {
    // 只有 utils.c 内部可以调用
}

void public_function(void) {
    private_helper();    // OK，同一文件内可调用
}
```

```c
// main.c
// static void private_helper(void);  // 无效！即使声明了也无法链接
// private_helper();                  // 链接错误：未定义的引用
```

## 三、模块封装——隐藏实现细节

### 3.1 模拟"私有"函数

C 语言没有 `private` 关键字，但 `static` 可以实现类似效果：

```c
// counter.c —— 计数器模块
#include <stdio.h>

static int count = 0;                          // 私有变量
static void validate(int value) {              // 私有函数
    if (value < 0) {
        fprintf(stderr, "计数不能为负\n");
    }
}

// 公开接口
void counter_init(void) {
    count = 0;
}

void counter_increment(void) {
    validate(count + 1);
    count++;
}

void counter_decrement(void) {
    validate(count - 1);
    if (count > 0) count--;
}

int counter_get(void) {
    return count;
}
```

### 3.2 头文件只暴露公开接口

```c
// counter.h —— 只声明公开的函数
#ifndef COUNTER_H
#define COUNTER_H

void counter_init(void);
void counter_increment(void);
void counter_decrement(void);
int  counter_get(void);

// 注意：没有 validate() 的声明——它是私有的
#endif
```

```c
// main.c
#include "counter.h"

int main() {
    counter_init();
    counter_increment();
    counter_increment();
    printf("计数: %d\n", counter_get());  // 2
    // validate(5);  // 链接错误！validate 是 static 的
    return 0;
}
```

## 四、static 的好处

### 4.1 避免命名冲突

```c
// module_a.c
static void init(void) { /* 模块A的初始化 */ }

// module_b.c
static void init(void) { /* 模块B的初始化 */ }
// 两个 init 函数不会冲突，因为各自只在本文件可见
```

### 4.2 明确接口边界

```c
// 读者看到 header 文件就知道哪些是公开接口
// 源文件中的 static 函数是实现细节，可以随时修改
```

### 4.3 编译器优化

```c
static int helper(int x) { return x * x; }

int compute(int a, int b) {
    return helper(a) + helper(b);
}

// 编译器知道 helper 只在本文件使用
// 可以更激进地内联和优化
```

## 五、static 变量 vs static 函数

```c
// static 修饰局部变量：延长生命周期到程序结束
void counter(void) {
    static int count = 0;  // 只初始化一次
    count++;
    printf("调用次数: %d\n", count);
}

// static 修饰全局变量：限制作用域到本文件
static int global_counter = 0;  // 只有本文件能访问

// static 修饰函数：限制作用域到本文件
static void helper(void) { }   // 只有本文件能调用
```

| 用途               | static 局部变量   | static 全局变量   | static 函数       |
|-------------------|------------------|------------------|------------------|
| 主要作用           | 延长生命周期       | 限制作用域        | 限制作用域         |
| 初始化次数         | 一次              | 程序启动时         | N/A              |
| 其他文件能否访问   | 不适用            | 否               | 否               |

## 六、常见使用模式

### 6.1 工具函数

```c
// string_utils.c
#include <string.h>
#include <ctype.h>

// 内部工具函数——不暴露给外部
static char* skip_whitespace(const char *s) {
    while (isspace((unsigned char)*s)) s++;
    return (char*)s;
}

static void to_lowercase(char *s) {
    for (; *s; s++) *s = tolower((unsigned char)*s);
}

// 公开接口
char* string_trim_and_lower(const char *input) {
    char *s = skip_whitespace(input);
    char *copy = strdup(s);
    to_lowercase(copy);
    return copy;
}
```

### 6.2 链表节点操作

```c
// linked_list.c
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// 私有：创建节点
static Node* create_node(int data) {
    Node *n = malloc(sizeof(Node));
    n->data = data;
    n->next = NULL;
    return n;
}

// 私有：查找节点
static Node* find_node(Node *head, int data) {
    while (head) {
        if (head->data == data) return head;
        head = head->next;
    }
    return NULL;
}

// 公开接口
void list_push(Node **head, int data) {
    Node *n = create_node(data);
    n->next = *head;
    *head = n;
}
```

## 七、要点总结

> **关键点：**
> 1. `static` 修饰函数使函数具有**内部链接**，只能在定义它的文件内访问。
> 2. 未加 `static` 的函数默认具有**外部链接**，可被其他文件通过 `extern` 调用。
> 3. 使用 `static` 函数实现**模块封装**，隐藏实现细节，只暴露必要接口。
> 4. `static` 可以避免不同模块之间的**命名冲突**。
> 5. 编译器可以对 `static` 函数进行更激进的优化（如内联）。
> 6. 良好的模块设计：头文件声明公开接口，源文件中用 `static` 保护私有函数。
