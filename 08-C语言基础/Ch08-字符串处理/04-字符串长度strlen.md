# 04 - 字符串长度 strlen

## 一、strlen 函数基本用法

### 1.1 函数原型

```c
#include <string.h>
size_t strlen(const char *str);
```

- **参数**：以 `\0` 结尾的字符串
- **返回值**：字符串长度（不包含 `\0`），类型为 `size_t`（无符号整数）
- **时间复杂度**：O(n)，需要遍历整个字符串

### 1.2 基本示例

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str1[] = "Hello";
    char str2[] = "";
    char str3[100] = "C Programming";

    printf("str1 长度: %zu\n", strlen(str1));    // 5
    printf("str2 长度: %zu\n", strlen(str2));    // 0
    printf("str3 长度: %zu\n", strlen(str3));    // 13

    // sizeof vs strlen
    printf("sizeof(str1): %zu\n", sizeof(str1)); // 6 (含\0)
    printf("strlen(str1): %zu\n", strlen(str1)); // 5 (不含\0)

    return 0;
}
```

## 二、strlen 的实现原理

### 2.1 基本实现

```c
// 经典实现
size_t my_strlen(const char *str) {
    const char *p = str;
    while (*p != '\0') {
        p++;
    }
    return (size_t)(p - str);
}
```

### 2.2 更简洁的写法

```c
size_t my_strlen2(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}
```

### 2.3 递归实现（仅供理解，不推荐）

```c
size_t my_strlen_recursive(const char *str) {
    if (*str == '\0') {
        return 0;
    }
    return 1 + my_strlen_recursive(str + 1);
}
```

## 三、strlen 与 sizeof 的区别

### 3.1 核心区别

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[20] = "Hello";

    // sizeof: 编译时计算，返回数组/类型占用的字节数
    printf("sizeof(str): %zu\n", sizeof(str));   // 20

    // strlen: 运行时计算，返回字符串实际长度
    printf("strlen(str): %zu\n", strlen(str));   // 5

    // 对于指针
    char *ptr = "Hello";
    printf("sizeof(ptr): %zu\n", sizeof(ptr));   // 8 (64位指针大小)
    printf("strlen(ptr): %zu\n", strlen(ptr));   // 5

    return 0;
}
```

### 3.2 对比表

| 特性 | sizeof | strlen |
|------|--------|--------|
| 本质 | 运算符 | 函数 |
| 计算时机 | 编译时 | 运行时 |
| 用途 | 计算占用字节数 | 计算字符串长度 |
| 含 `\0` | 包含 | 不包含 |
| 参数 | 类型或表达式 | 字符串指针 |
| 指针行为 | 返回指针大小 | 返回指向字符串长度 |

### 3.3 实际应用中的选择

```c
// 场景1：获取数组容量
char buf[100];
size_t capacity = sizeof(buf);  // ✅ 正确：100

// 场景2：获取字符串长度
char name[] = "Alice";
size_t len = strlen(name);      // ✅ 正确：5

// 场景3：计算需要复制的字符数
char src[] = "Hello";
char dst[10];
strncpy(dst, src, sizeof(dst) - 1);  // 使用sizeof获取缓冲区大小
dst[sizeof(dst) - 1] = '\0';
```

## 四、strlen 的使用注意事项

### 4.1 无符号整数陷阱

```c
#include <stdio.h>
#include <string.h>

int main() {
    char s1[] = "Hello";
    char s2[] = "Hi";

    // ⚠️ 危险：size_t 是无符号类型
    if (strlen(s1) - strlen(s2) > 0) {  // 永远为真！
        printf("This always prints!\n");
    }

    // ✅ 正确写法
    if (strlen(s1) > strlen(s2)) {
        printf("s1 is longer\n");
    }

    return 0;
}
```

### 4.2 必须有 \0 结尾

```c
// ⚠️ 未定义行为：没有 \0 终止符
char bad[] = {'H', 'e', 'l', 'l', 'o'};
size_t len = strlen(bad);  // 危险！会一直读取直到找到 \0

// ✅ 确保字符串以 \0 结尾
char good[] = {'H', 'e', 'l', 'l', 'o', '\0'};
```

### 4.3 性能问题

```c
// ❌ 低效：每次循环都调用 strlen
for (int i = 0; i < strlen(str); i++) {
    // O(n²) 时间复杂度！
}

// ✅ 高效：只调用一次
size_t len = strlen(str);
for (int i = 0; i < len; i++) {
    // O(n) 时间复杂度
}
```

## 五、扩展：计算字符串长度的其他方法

### 5.1 手动遍历

```c
size_t string_length(const char *s) {
    size_t count = 0;
    while (*s++) count++;
    return count;
}
```

### 5.2 使用 strchr

```c
#include <string.h>

size_t length_via strchr(const char *s) {
    const char *end = strchr(s, '\0');
    return (size_t)(end - s);
}
```

### 5.3 多字节字符串长度（UTF-8）

```c
#include <stdio.h>

// UTF-8字符串的字符数（不是字节数）
size_t utf8_char_count(const char *s) {
    size_t count = 0;
    while (*s) {
        // 跳过UTF-8续字节 (10xxxxxx)
        if ((*s & 0xC0) != 0x80) {
            count++;
        }
        s++;
    }
    return count;
}

int main() {
    char *text = "你好World";
    printf("字节数: %zu\n", strlen(text));         // 可能是 11
    printf("字符数: %zu\n", utf8_char_count(text)); // 7
    return 0;
}
```

## 六、重要注意事项

> **关键要点：**
> 1. `strlen` 返回 `size_t`（无符号），与有符号数比较时要小心
> 2. `strlen` 时间复杂度为 O(n)，避免在循环条件中重复调用
> 3. `strlen` 和 `sizeof` 是完全不同的概念，不要混淆
> 4. 传给 `strlen` 的字符串必须以 `\0` 结尾，否则未定义行为
> 5. 空字符串 `""` 的 `strlen` 返回 0
> 6. `strlen` 不包含 `\0` 在长度中
