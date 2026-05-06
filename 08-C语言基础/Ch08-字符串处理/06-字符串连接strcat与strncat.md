# 06 - 字符串连接 strcat 与 strncat

## 一、strcat 基本用法

### 1.1 函数原型

```c
#include <string.h>
char *strcat(char *dest, const char *src);
```

- 将 `src` 追加到 `dest` 末尾（覆盖 `dest` 原来的 `\0`）
- 在结果末尾添加 `\0`
- 返回值：`dest` 指针
- **不检查目标缓冲区剩余空间**

### 1.2 基本示例

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[50] = "Hello";

    strcat(str, ", ");
    strcat(str, "World!");

    printf("%s\n", str);  // Hello, World!

    // 链式调用
    char buf[100] = "";
    strcat(buf, "C ");
    strcat(buf, "is ");
    strcat(buf, "powerful!");
    printf("%s\n", buf);  // C is powerful!

    return 0;
}
```

### 1.3 strcat 的实现原理

```c
char *my_strcat(char *dest, const char *src) {
    char *d = dest;

    // 1. 找到 dest 的末尾 \0
    while (*d != '\0') {
        d++;
    }

    // 2. 将 src 复制到 dest 末尾
    while ((*d++ = *src++) != '\0') {
        ;
    }

    return dest;
}
```

**执行过程：**

```
dest:  H  e  l  l  o  \0
              ↓
src:   ,     W  o  r  l  d  \0
       ↓     ↓  ↓  ↓  ↓  ↓

结果:  H  e  l  l  o  ,     W  o  r  l  d  \0
```

## 二、strcat 的危险性

### 2.1 缓冲区溢出

```c
// ⚠️ 危险示例
char str[10] = "Hello";
strcat(str, ", World!");  // 溢出！
// str 只有10字节，"Hello" + ", World!" = 13字符 + \0 = 14字节
// 结果：未定义行为
```

### 2.2 常见错误

```c
// 错误1：未初始化的目标
char dest[50];
strcat(dest, "Hello");  // 危险！dest未初始化，\0位置未知

// 正确做法
char dest[50] = "";
strcat(dest, "Hello");

// 错误2：目标是指向字面量的指针
char *dest = "Hello";
strcat(dest, " World");  // 未定义行为！修改只读内存

// 错误3：自连接
char str[20] = "abc";
strcat(str, str);  // 未定义行为！源和目标重叠
```

## 三、strncat 安全连接

### 3.1 函数原型

```c
#include <string.h>
char *strncat(char *dest, const char *src, size_t n);
```

- 最多从 `src` 追加 `n` 个字符到 `dest`
- **始终在末尾添加 `\0`**
- 如果 `src` 长度小于 `n`，只追加到 `\0` 为止

### 3.2 基本示例

```c
#include <stdio.h>
#include <string.h>

int main() {
    char dest[20] = "Hello";

    // 最多追加5个字符
    strncat(dest, ", World!", 5);
    printf("%s\n", dest);  // Hello, Wor

    // 继续追加
    strncat(dest, "ld!", 10);
    printf("%s\n", dest);  // Hello, World!

    return 0;
}
```

### 3.3 strncat 的安全使用模式

```c
#include <stdio.h>
#include <string.h>

// 安全连接：自动计算剩余空间
char *safe_strcat(char *dest, const char *src, size_t dest_size) {
    size_t dest_len = strlen(dest);
    size_t remaining = dest_size - dest_len - 1;  // 预留\0的空间

    if (remaining > 0) {
        strncat(dest, src, remaining);
    }

    return dest;
}

int main() {
    char buf[20] = "Hello";

    safe_strcat(buf, ", World! This is too long", sizeof(buf));
    printf("%s\n", buf);  // Hello, World! This

    return 0;
}
```

## 四、多次连接的效率问题

### 4.1 低效的重复连接

```c
// ❌ 低效：每次 strcat 都要遍历到末尾
char result[1000] = "";
for (int i = 0; i < 1000; i++) {
    strcat(result, "a");  // O(n) 每次，总 O(n²)
}
```

### 4.2 高效的连接方式

```c
// ✅ 高效：维护一个指针指向末尾
char result[1000];
char *end = result;
*end = '\0';

for (int i = 0; i < 1000; i++) {
    *end++ = 'a';
}
*end = '\0';

// 或使用 sprintf/snprintf 一次性连接
char buf[200];
snprintf(buf, sizeof(buf), "%s%s%s", "Hello", ", ", "World");
```

## 五、strcat 与 strncat 对比

| 特性 | strcat | strncat |
|------|--------|---------|
| 安全性 | 不安全 | 较安全 |
| 检查长度 | 否 | 是 |
| 自动加\0 | 是 | 是（始终添加）|
| 性能 | 遍历到末尾 | 遍历到末尾 |

## 六、重要注意事项

> **关键要点：**
> 1. **目标必须以 `\0` 结尾**：否则 `strcat` 无法找到追加位置
> 2. **目标必须有足够空间**：至少 `strlen(dest) + strlen(src) + 1`
> 3. **`strncat` 始终添加 `\0`**：与 `strncpy` 不同
> 4. **计算剩余空间时别忘了减1**：为 `\0` 预留位置
> 5. **避免频繁连接**：大量连接操作时，考虑维护尾指针或使用 `snprintf`
> 6. **源和目标不能重叠**：重叠行为未定义
