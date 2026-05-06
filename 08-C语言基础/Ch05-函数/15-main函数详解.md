# 15 - main 函数详解

## 一、main 函数的特殊地位

`main` 函数是 C 程序的**入口点**。操作系统加载程序后，首先调用 `main` 函数开始执行。

```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

## 二、main 的两种标准形式

### 2.1 无命令行参数

```c
int main(void) {
    // 不需要命令行参数
    return 0;
}
```

### 2.2 带命令行参数

```c
int main(int argc, char *argv[]) {
    // argc: 参数个数（包括程序名）
    // argv: 参数字符串数组
    return 0;
}
```

> **注意：** 标准只规定了以上两种形式。有些编译器还支持 `char *envp[]`（环境变量），但这不属于标准。

## 三、返回值

### 3.1 返回值的含义

`main` 函数的返回值是一个**退出状态码**，传递给操作系统：

```c
int main(void) {
    // return 0; 表示程序正常结束
    // return 非0; 表示程序异常结束
    return 0;
}
```

| 返回值 | 含义                 |
|-------|---------------------|
| 0     | 程序成功执行          |
| 非0   | 程序执行出错          |

### 3.2 不同返回值的用法

```c
int main(void) {
    FILE *f = fopen("data.txt", "r");
    if (!f) {
        fprintf(stderr, "无法打开文件\n");
        return 1;   // 返回 1 表示出错
    }
    // ...
    fclose(f);
    return 0;       // 返回 0 表示成功
}
```

### 3.3 在 shell 中检查返回值

```bash
./myprogram
echo $?    # 打印上一个程序的退出码

# 在脚本中使用
if ./myprogram; then
    echo "成功"
else
    echo "失败，退出码: $?"
fi
```

### 3.4 return 0 与 exit(0)

```c
int main(void) {
    // 方式一：return 语句
    return 0;

    // 方式二：exit 函数（效果相同）
    // exit(0);

    // 方式三：省略 return（C99 起隐式返回 0）
    // 不推荐，代码意图不清晰
}
```

> **C99 规则：** 如果 `main` 函数执行到末尾没有 `return`，等价于 `return 0;`。但建议始终显式写出。

## 四、命令行参数 argc 和 argv

### 4.1 argc（Argument Count）

`argc` 是命令行参数的个数，**包括程序名本身**，所以至少为 1。

```bash
./myprogram hello world
# argc = 3（"./myprogram", "hello", "world"）
```

### 4.2 argv（Argument Vector）

`argv` 是一个字符串指针数组，每个元素指向一个命令行参数：

```
argv[0] -> "./myprogram"
argv[1] -> "hello"
argv[2] -> "world"
argv[3] -> NULL  （最后一个元素总是 NULL）
```

### 4.3 基本示例

```c
#include <stdio.h>

int main(int argc, char *argv[]) {
    printf("参数个数: %d\n", argc);

    for (int i = 0; i < argc; i++) {
        printf("argv[%d] = \"%s\"\n", i, argv[i]);
    }

    return 0;
}
```

运行：
```
$ ./myprogram hello world 123
参数个数: 4
argv[0] = "./myprogram"
argv[1] = "hello"
argv[2] = "world"
argv[3] = "123"
```

### 4.4 简单的计算器

```c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "用法: %s 数字 运算符 数字\n", argv[0]);
        return 1;
    }

    double a = atof(argv[1]);
    char op = argv[2][0];
    double b = atof(argv[3]);

    switch (op) {
        case '+': printf("%.2f\n", a + b); break;
        case '-': printf("%.2f\n", a - b); break;
        case '*': printf("%.2f\n", a * b); break;
        case '/':
            if (b == 0) { printf("错误：除零\n"); return 1; }
            printf("%.2f\n", a / b); break;
        default:
            fprintf(stderr, "未知运算符: %c\n", op);
            return 1;
    }

    return 0;
}
```

```bash
./calc 3 + 5      # 输出: 8.00
./calc 10 / 3     # 输出: 3.33
```

## 五、argv 的内存布局

```
argv 指针数组          字符串数据
+-----------+
| argv[0]   | -------> "./myprogram\0"
+-----------+
| argv[1]   | -------> "hello\0"
+-----------+
| argv[2]   | -------> "world\0"
+-----------+
| argv[3]   | -------> NULL
+-----------+
```

### 5.1 修改 argv 的内容

```c
// argv 指向的字符串可以修改（但字符串字面量不行）
int main(int argc, char *argv[]) {
    for (int i = 0; i < argc; i++) {
        char *p = argv[i];
        while (*p) {
            *p = toupper(*p);  // 转换为大写
            p++;
        }
        printf("%s\n", argv[i]);
    }
    return 0;
}
```

## 六、main 函数的等价形式

```c
// 标准形式一
int main(void) { return 0; }

// 标准形式二
int main(int argc, char *argv[]) { return 0; }

// 等价写法（argv 可以写成 char **）
int main(int argc, char **argv) { return 0; }

// 常见但非标准的扩展
int main(int argc, char *argv[], char *envp[]) { return 0; }
```

## 七、要点总结

> **关键点：**
> 1. `main` 是程序的入口点，标准规定只有两种形式。
> 2. 返回值表示退出状态码，0 表示成功，非 0 表示错误。
> 3. `argc` 包括程序名本身，至少为 1。
> 4. `argv[argc]` 总是 `NULL`。
> 5. `argv[0]` 通常是程序名，可用于显示用法信息。
> 6. 命令行参数都是字符串，需要手动转换为数字（`atoi`, `atof`）。
> 7. C99 起 `main` 末尾省略 `return` 等价于 `return 0;`，但建议显式写出。
