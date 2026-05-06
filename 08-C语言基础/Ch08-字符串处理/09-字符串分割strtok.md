# 09 - 字符串分割 strtok

## 一、strtok 基本用法

### 1.1 函数原型

```c
#include <string.h>
char *strtok(char *str, const char *delim);
```

- 将字符串按分隔符拆分为多个"令牌"（token）
- **首次调用**传入待分割字符串
- **后续调用**传入 `NULL`，继续上次分割
- 找到令牌返回其指针，无更多令牌返回 `NULL`

### 1.2 基本示例

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "Hello,World,C,Language";

    // 首次调用
    char *token = strtok(str, ",");
    while (token != NULL) {
        printf("%s\n", token);
        token = strtok(NULL, ",");  // 后续调用传NULL
    }
    // 输出:
    // Hello
    // World
    // C
    // Language

    return 0;
}
```

### 1.3 执行过程

```
原字符串: "Hello,World,C,Language"

第1次 strtok(str, ","):
  找到第一个 ',' → 替换为 \0
  返回: "Hello"

第2次 strtok(NULL, ","):
  从上个 \0 之后开始
  找到下一个 ',' → 替换为 \0
  返回: "World"

第3次 strtok(NULL, ","):
  返回: "C"

第4次 strtok(NULL, ","):
  返回: "Language"

第5次 strtok(NULL, ","):
  返回: NULL（结束）
```

## 二、多分隔符

### 2.1 使用多个分隔符

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "Hello, World; C\tLanguage\nProgramming";

    // 使用逗号、空格、分号、Tab、换行作为分隔符
    char *token = strtok(str, ", ;\t\n");
    while (token != NULL) {
        printf("'%s'\n", token);
        token = strtok(NULL, ", ;\t\n");
    }
    // 'Hello'
    // 'World'
    // 'C'
    // 'Language'
    // 'Programming'

    return 0;
}
```

### 2.2 处理连续分隔符

```c
#include <stdio.h>
#include <string.h>

int main() {
    // 连续分隔符被视为一个分隔符
    char str[] = "a,,b,,,c";
    char *token = strtok(str, ",");
    while (token != NULL) {
        printf("'%s' ", token);
        token = strtok(NULL, ",");
    }
    // 输出: 'a' 'b' 'c'
    // 注意：空字段被跳过

    return 0;
}
```

## 三、strtok 的限制与陷阱

### 3.1 修改原字符串

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "Hello,World";
    char *token = strtok(str, ",");

    // ⚠️ 原字符串已被修改！
    printf("原字符串: %s\n", str);  // "Hello" (逗号被\0替代)

    // 不能对 const 字符串使用
    // char *s = "Hello,World";
    // strtok(s, ",");  // 未定义行为！修改只读内存

    return 0;
}
```

### 3.2 不可重入（非线程安全）

```c
// ⚠️ strtok 使用内部静态变量保存状态
// 多线程环境下会有问题！

// 线程A中:
token = strtok(str1, ",");

// 线程B中（同时执行）:
token = strtok(str2, ",");
// 会干扰线程A的状态！

// 解决方案：使用 strtok_r（可重入版本）
// char *strtok_r(char *str, const char *delim, char **saveptr);
```

### 3.3 strtok_r 可重入版本

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "a,b,c,d,e";
    char *saveptr;

    char *token = strtok_r(str, ",", &saveptr);
    while (token != NULL) {
        printf("%s\n", token);
        token = strtok_r(NULL, ",", &saveptr);
    }

    // 可以同时分割两个字符串
    char s1[] = "1,2,3";
    char s2[] = "a,b,c";
    char *save1, *save2;

    char *t1 = strtok_r(s1, ",", &save1);
    char *t2 = strtok_r(s2, ",", &save2);
    while (t1 && t2) {
        printf("%s - %s\n", t1, t2);
        t1 = strtok_r(NULL, ",", &save1);
        t2 = strtok_r(NULL, ",", &save2);
    }
    // 1 - a
    // 2 - b
    // 3 - c

    return 0;
}
```

## 四、实际应用示例

### 4.1 解析CSV数据

```c
#include <stdio.h>
#include <string.h>

int main() {
    char line[] = "张三,90,85,92";

    char *name = strtok(line, ",");
    char *score1 = strtok(NULL, ",");
    char *score2 = strtok(NULL, ",");
    char *score3 = strtok(NULL, ",");

    if (name && score1 && score2 && score3) {
        printf("姓名: %s\n", name);
        printf("成绩: %s, %s, %s\n", score1, score2, score3);
    }

    return 0;
}
```

### 4.2 解析命令行参数

```c
#include <stdio.h>
#include <string.h>

int main() {
    char input[] = "copy file1.txt file2.txt";
    char *args[10];
    int argc = 0;

    char *token = strtok(input, " ");
    while (token != NULL && argc < 10) {
        args[argc++] = token;
        token = strtok(NULL, " ");
    }

    printf("命令: %s\n", args[0]);
    printf("参数数量: %d\n", argc - 1);
    for (int i = 1; i < argc; i++) {
        printf("  参数%d: %s\n", i, args[i]);
    }

    return 0;
}
```

## 五、重要注意事项

> **关键要点：**
> 1. **`strtok` 会修改原字符串**：不能用于 `const` 或字面量字符串
> 2. **`strtok` 非线程安全**：多线程使用 `strtok_r`
> 3. **首次调用传字符串，后续传 NULL**
> 4. **连续分隔符被视为一个**：空字段会被跳过
> 5. **返回的指针指向原字符串内部**：不需要 `free`，但原字符串必须保持有效
> 6. **想要保留空字段**：需要自己实现分割函数
