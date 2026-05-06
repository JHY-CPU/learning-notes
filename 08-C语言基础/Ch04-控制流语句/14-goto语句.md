# goto 语句

## 1. goto 概述

`goto` 是C语言中唯一的无条件跳转语句，可以将程序控制权直接转移到函数内的任意标签处。虽然 goto 功能强大，但在现代编程中被普遍认为应该尽量避免使用。

## 2. 语法

```c
goto 标签名;
// ...
标签名:
    语句;
```

**规则：**
- 标签名是一个标识符，后面跟一个冒号
- goto 和标签必须在**同一个函数内**
- 不能跨函数跳转
- 标签的作用范围是整个函数

## 3. 基本示例

```c
#include <stdio.h>

int main(void) {
    int i = 0;

start:                          // 标签
    if (i >= 5) {
        goto end;               // 跳转到 end 标签
    }
    printf("%d ", i);
    i++;
    goto start;                 // 跳转回 start 标签

end:                            // 标签
    printf("\n结束\n");

    return 0;
}
// 输出: 0 1 2 3 4
//      结束
```

## 4. goto 的合理使用场景

### 4.1 错误处理和资源清理

这是 goto **最被接受的使用场景**。当多层嵌套的操作中某一步失败时，goto 可以优雅地跳转到统一的清理代码：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int process_file(const char *filename) {
    FILE *fp = NULL;
    char *buffer = NULL;
    int *data = NULL;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        goto error;
    }

    buffer = malloc(1024);
    if (buffer == NULL) {
        goto error;
    }

    data = malloc(sizeof(int) * 100);
    if (data == NULL) {
        goto error;
    }

    // 正常处理...
    printf("处理成功\n");

    // 正常清理
    free(data);
    free(buffer);
    fclose(fp);
    return 0;

error:
    // 统一的错误处理和资源清理
    printf("发生错误\n");
    if (data) free(data);
    if (buffer) free(buffer);
    if (fp) fclose(fp);
    return -1;
}
```

**不使用 goto 的等价写法**（深层嵌套）：

```c
int process_file_no_goto(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp != NULL) {
        char *buffer = malloc(1024);
        if (buffer != NULL) {
            int *data = malloc(sizeof(int) * 100);
            if (data != NULL) {
                // 处理...
                free(data);
                free(buffer);
                fclose(fp);
                return 0;
            } else {
                free(buffer);
                fclose(fp);
                return -1;
            }
        } else {
            fclose(fp);
            return -1;
        }
    }
    return -1;
}
```

可以看到，不用 goto 的版本嵌套更深，重复的清理代码更多。

### 4.2 跳出多层嵌套循环

```c
#include <stdio.h>

int main(void) {
    int matrix[3][4] = {
        {1, 5, 3, 8},
        {2, 7, 4, 6},
        {9, 0, 3, 5}
    };

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            if (matrix[i][j] == 7) {
                printf("找到 7 在 [%d][%d]\n", i, j);
                goto found;    // 直接跳出两层循环
            }
        }
    }
    printf("未找到\n");

found:
    printf("搜索结束\n");
    return 0;
}
```

### 4.3 中断深度嵌套的处理

```c
int validate_and_process(Input *input) {
    if (!check_format(input)) {
        goto invalid;
    }

    if (!check_permissions(input)) {
        goto unauthorized;
    }

    if (!check_resources(input)) {
        goto insufficient;
    }

    // 处理成功
    do_process(input);
    return SUCCESS;

invalid:
    log_error("格式无效");
    return ERR_FORMAT;

unauthorized:
    log_error("权限不足");
    return ERR_PERMISSION;

insufficient:
    log_error("资源不足");
    return ERR_RESOURCE;
}
```

## 5. goto 的危害

### 5.1 破坏结构化编程

```c
// 难以理解的跳转
int x = 0;
start:
    x++;
    if (x % 2 == 0) goto even;
    printf("奇数: %d\n", x);
    if (x < 10) goto start;
    goto end;
even:
    printf("偶数: %d\n", x);
    x++;
    goto start;
end:
    printf("结束\n");
```

这段代码虽然正确，但逻辑极难跟踪。

### 5.2 使控制流不可预测

```c
// 使用 goto 的代码——流程难以跟踪
if (condition1) goto label_a;
if (condition2) goto label_b;
// ...
label_a:
    // ...
    if (condition3) goto label_c;
    goto label_b;
label_b:
    // ...
label_c:
    // ...
```

### 5.3 难以调试和维护

- goto 跳转使得程序执行路径不线性
- 编译器优化受到限制
- 代码审查更困难

## 6. goto 的替代方案

### 6.1 使用函数

```c
// 将复杂逻辑分解为小函数
int validate_input(Input *in);
int check_permissions(Input *in);
int process_data(Input *in);

int handle(Input *in) {
    if (!validate_input(in)) return ERR_FORMAT;
    if (!check_permissions(in)) return ERR_PERMISSION;
    return process_data(in);
}
```

### 6.2 使用 break/continue

```c
// 用 break 替代 goto 跳出循环
for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        if (found) break;  // 只能跳出一层
    }
    if (found) break;      // 需要额外判断
}
```

### 6.3 使用 do-while(0) 惯用法

```c
do {
    if (!step1()) break;
    if (!step2()) break;
    if (!step3()) break;
    success = 1;
} while (0);

if (!success) {
    cleanup();
}
```

## 7. goto 的使用规范

如果必须使用 goto，请遵循以下规范：

1. **只向下跳转**：不要向前跳转（不要形成循环）
2. **只用于错误处理**：goto 的主要合理用途是集中清理
3. **标签命名清晰**：如 `error_open`、`error_alloc` 等
4. **限制使用范围**：只在函数开头或错误处理段使用
5. **加注释说明**：让读者理解跳转的原因

```c
// 符合规范的 goto 使用
int allocate_resources(Config *cfg) {
    Resource *r1 = NULL;
    Resource *r2 = NULL;
    Resource *r3 = NULL;

    r1 = alloc_resource(cfg->size1);
    if (!r1) goto error_alloc_r1;

    r2 = alloc_resource(cfg->size2);
    if (!r2) goto error_alloc_r2;

    r3 = alloc_resource(cfg->size3);
    if (!r3) goto error_alloc_r3;

    return SUCCESS;

error_alloc_r3:
    free_resource(r2);
error_alloc_r2:
    free_resource(r1);
error_alloc_r1:
    return ERR_ALLOCATION;
}
```

注意清理顺序是分配的逆序（后分配的先释放），这是经典的资源管理模式。

## 8. 要点总结

1. `goto` 是无条件跳转，可以在函数内任意跳转
2. **唯一被广泛接受的用途**：错误处理和资源清理
3. 滥用 goto 会破坏结构化编程，使代码难以理解和维护
4. 大多数 goto 场景可以用函数分解、break、do-while(0) 替代
5. 如果使用 goto，只向下跳转，用于错误清理，并加注释
6. Linux 内核中 goto 被大量使用，主要就是用于错误处理

## 9. 练习题

1. 用 goto 实现一个多步骤操作的错误处理（每步失败时清理已分配的资源）
2. 将上面的 goto 代码改写为不使用 goto 的等价版本
3. 思考：在什么情况下，使用 goto 的代码比不使用 goto 的更清晰？
