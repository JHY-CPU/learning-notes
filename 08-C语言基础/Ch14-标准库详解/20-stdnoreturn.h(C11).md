# stdnoreturn.h - _Noreturn 说明符（C11）

## 1. 概述

`<stdnoreturn.h>`（C11引入）定义了 `noreturn` 宏，用于标记永远不会返回的函数。这可以帮助编译器进行更好的优化和错误检测。

## 2. 核心定义

```c
#include <stdnoreturn.h>

#define noreturn _Noreturn

// C23中 _Noreturn 成为 [[noreturn]] 属性
```

## 3. 基本用法

### 3.1 标记不返回的函数

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>

// 这个函数永远不会返回到调用者
noreturn void fatal_error(const char *msg) {
    fprintf(stderr, "致命错误: %s\n", msg);
    exit(EXIT_FAILURE);
    // 编译器知道这里不需要return语句
}

// 另一个不返回的函数
noreturn void infinite_loop(void) {
    while (1) {
        // 永不停止
    }
    // 编译器不会警告缺少return
}

int main(void) {
    printf("程序开始\n");

    // 触发致命错误
    // fatal_error("测试错误");

    printf("程序正常结束\n");
    return 0;
}
```

### 3.2 常见的 noreturn 函数

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>

// 标准库中的noreturn函数：
// void exit(int status);
// void _Exit(int status);
// void abort(void);
// void longjmp(jmp_buf, int); （在某些实现中）

// 自定义错误处理
noreturn void panic(const char *file, int line, const char *msg) {
    fprintf(stderr, "PANIC at %s:%d: %s\n", file, line, msg);
    abort();
}

// 方便的宏
#define PANIC(msg) panic(__FILE__, __LINE__, msg)

// 不返回的断言
noreturn void assert_fail(const char *expr, const char *file, int line) {
    fprintf(stderr, "断言失败: %s (%s:%d)\n", expr, file, line);
    abort();
}

int main(void) {
    // 使用示例
    int *ptr = malloc(sizeof(int));
    if (ptr == NULL) {
        PANIC("内存分配失败");
    }
    free(ptr);

    return 0;
}
```

## 4. 编译器优化

```c
#include <stdio.h>
#include <stdnoreturn.h>
#include <stdlib.h>

noreturn void die(void) {
    exit(1);
}

// 编译器可以利用noreturn信息进行优化
int example(int x) {
    if (x > 0) {
        return x * 2;
    }
    die();  // 编译器知道这里不会返回
    // 没有noreturn时，编译器可能警告缺少return语句
    // 有noreturn时，编译器知道控制流不会到达这里
}

// switch语句中的应用
const char *color_name(int code) {
    switch (code) {
        case 0: return "红色";
        case 1: return "绿色";
        case 2: return "蓝色";
        default:
            PANIC("无效的颜色代码");
            // 编译器不需要为default分支生成return
    }
}

int main(void) {
    printf("颜色: %s\n", color_name(0));
    // printf("无效: %s\n", color_name(99));  // 触发PANIC
    return 0;
}
```

## 5. 重要注意事项

> **要点一**：`noreturn` 函数如果实际返回了，行为是未定义的。

> **要点二**：`noreturn` 是对编译器的提示，帮助消除警告和进行优化。

> **要点三**：标准库函数 `exit`、`_Exit`、`abort` 和 `longjmp` 都是 noreturn 的。

> **要点四**：`noreturn` 只能用于函数声明，不能用于函数指针的类型（在某些编译器中）。

> **要点五**：C23中 `_Noreturn` 将被 `[[noreturn]]` 属性取代。

> **要点六**：如果函数可能在某些条件下不返回，在其他条件下返回，则不应标记为 `noreturn`。

> **要点七**：使用 `noreturn` 可以消除"控制流到达非void函数结尾"的编译器警告。
