# 05 - 字符串复制 strcpy 与 strncpy

## 一、strcpy 基本用法

### 1.1 函数原型

```c
#include <string.h>
char *strcpy(char *dest, const char *src);
```

- 将 `src` 指向的字符串（含 `\0`）复制到 `dest`
- 返回值：`dest` 指针
- **不检查目标缓冲区大小**，使用不当会导致缓冲区溢出

### 1.2 基本示例

```c
#include <stdio.h>
#include <string.h>

int main() {
    char src[] = "Hello, World!";
    char dest[50];

    strcpy(dest, src);

    printf("源字符串: %s\n", src);      // Hello, World!
    printf("目标字符串: %s\n", dest);    // Hello, World!

    // strcpy 返回目标指针，可用于链式调用
    char buf[100];
    printf("%s\n", strcpy(buf, "Test")); // Test

    return 0;
}
```

### 1.3 strcpy 的实现原理

```c
char *my_strcpy(char *dest, const char *src) {
    char *d = dest;           // 保存目标地址
    while ((*d++ = *src++) != '\0') {
        ;  // 复制每个字符，包括 \0
    }
    return dest;
}
```

**执行过程演示：**

```
src: H  e  l  l  o  \0
     ↓  ↓  ↓  ↓  ↓  ↓
dest: H  e  l  l  o  \0 ...
```

## 二、strcpy 的危险性

### 2.1 缓冲区溢出

```c
// ⚠️ 危险代码
char small[5];
char *long_str = "Hello, World!";  // 13个字符 + \0

strcpy(small, long_str);  // 缓冲区溢出！只分配了5字节
// 结果：未定义行为，可能覆盖相邻内存，程序崩溃或安全漏洞
```

### 2.2 常见错误

```c
// 错误1：目标未分配内存
char *dest;
strcpy(dest, "Hello");    // 崩溃！dest未指向有效内存

// 错误2：目标是字符串字面量
char *dest = "existing";
strcpy(dest, "new");      // 未定义行为！修改字面量

// 错误3：源和目标重叠
char str[20] = "Hello";
strcpy(str + 1, str);     // 未定义行为！内存重叠
```

## 三、strncpy 安全复制

### 3.1 函数原型

```c
#include <string.h>
char *strncpy(char *dest, const char *src, size_t n);
```

- 最多复制 `n` 个字符到 `dest`
- 如果 `src` 长度小于 `n`，剩余部分填充 `\0`
- 如果 `src` 长度大于等于 `n`，**不会自动添加 `\0`**

### 3.2 基本示例

```c
#include <stdio.h>
#include <string.h>

int main() {
    char dest[10];

    // 安全复制：最多复制9个字符，留1位给\0
    strncpy(dest, "Hello, World!", sizeof(dest) - 1);
    dest[sizeof(dest) - 1] = '\0';  // 手动确保\0结尾

    printf("dest: %s\n", dest);  // Hello, Wor

    return 0;
}
```

### 3.3 strncpy 的行为详解

```c
char dest[10];

// 情况1：src 长度 < n
strncpy(dest, "Hi", 10);
// dest: H i \0 \0 \0 \0 \0 \0 \0 \0
// 多余位置用 \0 填充

// 情况2：src 长度 >= n
strncpy(dest, "Hello, World!", 10);
// dest: H e l l o ,   W o   (没有 \0!)
// ⚠️ 必须手动添加 \0
```

## 四、安全的字符串复制实践

### 4.1 封装安全复制函数

```c
#include <string.h>

// 安全复制：确保目标以\0结尾
char *safe_strcpy(char *dest, const char *src, size_t dest_size) {
    if (dest_size == 0) return dest;

    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';

    return dest;
}
```

### 4.2 使用 strlcpy（部分平台支持）

```c
// BSD/macOS 提供 strlcpy
// size_t strlcpy(char *dest, const char *src, size_t dest_size);
// 总是以\0结尾，返回源字符串长度

// Windows: 使用 strcpy_s
// errno_t strcpy_s(char *dest, rsize_t destsz, const char *src);

// C11: 使用 strcpy_s (Annex K)
// strcpy_s(dest, sizeof(dest), src);
```

### 4.3 完整示例：用户输入复制

```c
#include <stdio.h>
#include <string.h>

#define MAX_NAME 32

int main() {
    char name[MAX_NAME];
    char greeting[MAX_NAME + 20];

    printf("请输入姓名: ");
    if (fgets(name, sizeof(name), stdin)) {
        // 去除换行符
        name[strcspn(name, "\n")] = '\0';

        // 安全复制和拼接
        safe_strcpy(greeting, "你好, ", sizeof(greeting));
        strncat(greeting, name, sizeof(greeting) - strlen(greeting) - 1);

        printf("%s\n", greeting);
    }

    return 0;
}
```

## 五、strcpy 与 strncpy 对比

| 特性 | strcpy | strncpy |
|------|--------|---------|
| 安全性 | 不安全 | 较安全 |
| 检查长度 | 否 | 是 |
| 自动加\0 | 是 | 不一定 |
| 性能 | 较快 | 较慢 |
| 填充行为 | 无 | 不足时补\0 |

## 六、重要注意事项

> **关键要点：**
> 1. **`strcpy` 不检查目标缓冲区大小**，极易造成缓冲区溢出
> 2. **`strncpy` 不保证添加 `\0`**：当源字符串长度 >= n 时，需要手动添加
> 3. **推荐模式**：`strncpy + 手动\0` 或封装安全函数
> 4. **目标必须有足够的空间**：至少 `strlen(src) + 1` 字节
> 5. **源和目标内存不能重叠**：重叠行为未定义（使用 `memmove` 处理重叠）
> 6. **返回值可以链式调用**：`printf("%s\n", strcpy(buf, "test"));`
> 7. 现代编译器会对 `strcpy` 发出警告，建议使用安全替代
