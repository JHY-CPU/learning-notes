# 07 - 字符串比较 strcmp 与 strncmp

## 一、strcmp 基本用法

### 1.1 函数原型

```c
#include <string.h>
int strcmp(const char *s1, const char *s2);
```

- 按字典序（ASCII值）逐字符比较两个字符串
- **返回值**：
  - `0`：两个字符串相等
  - `< 0`：`s1` 小于 `s2`
  - `> 0`：`s1` 大于 `s2`

### 1.2 基本示例

```c
#include <stdio.h>
#include <string.h>

int main() {
    // 相等
    printf("%d\n", strcmp("Hello", "Hello"));    // 0

    // s1 < s2（'e' < 'o'）
    printf("%d\n", strcmp("Hello", "World"));    // 负数

    // s1 > s2（'W' > 'H'）
    printf("%d\n", strcmp("World", "Hello"));    // 正数

    // 前缀关系
    printf("%d\n", strcmp("Hello", "Hello, World")); // 负数（'\0' < ','）

    // 大小写敏感
    printf("%d\n", strcmp("hello", "Hello"));    // 正数（'h' > 'H'）

    return 0;
}
```

### 1.3 strcmp 的实现原理

```c
int my_strcmp(const char *s1, const char *s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return (unsigned char)*s1 - (unsigned char)*s2;
}
```

**比较过程演示：**

```
s1: H  e  l  l  o
s2: H  e  l  p
         ↓
比较到第4个字符: 'l' vs 'p'
返回 'l' - 'p' = 108 - 112 = -4
```

## 二、strncmp 限定长度比较

### 2.1 函数原型

```c
#include <string.h>
int strncmp(const char *s1, const char *s2, size_t n);
```

- 最多比较前 `n` 个字符
- 如果前 `n` 个字符都相等，返回 `0`
- 遇到 `\0` 也会停止比较

### 2.2 基本示例

```c
#include <stdio.h>
#include <string.h>

int main() {
    // 只比较前5个字符
    printf("%d\n", strncmp("Hello World", "Hello C", 5));   // 0

    // 前3个字符不同
    printf("%d\n", strncmp("Hello", "World", 3));            // 负数

    // 检查前缀
    if (strncmp("http://example.com", "http://", 7) == 0) {
        printf("这是HTTP链接\n");
    }

    // 检查文件扩展名
    char filename[] = "document.pdf";
    if (strncmp(filename + strlen(filename) - 4, ".pdf", 4) == 0) {
        printf("这是一个PDF文件\n");
    }

    return 0;
}
```

## 三、字符串比较的实际应用

### 3.1 用户输入验证

```c
#include <stdio.h>
#include <string.h>

int main() {
    char password[50];
    const char *correct = "secret123";

    printf("请输入密码: ");
    scanf("%49s", password);

    if (strcmp(password, correct) == 0) {
        printf("密码正确！\n");
    } else {
        printf("密码错误！\n");
    }

    return 0;
}
```

### 3.2 菜单选项匹配

```c
#include <stdio.h>
#include <string.h>

int main() {
    char cmd[20];

    printf("请输入命令 (start/stop/restart): ");
    scanf("%19s", cmd);

    if (strcmp(cmd, "start") == 0) {
        printf("启动服务...\n");
    } else if (strcmp(cmd, "stop") == 0) {
        printf("停止服务...\n");
    } else if (strcmp(cmd, "restart") == 0) {
        printf("重启服务...\n");
    } else {
        printf("未知命令: %s\n", cmd);
    }

    return 0;
}
```

### 3.3 大小写不敏感比较

```c
#include <stdio.h>
#include <ctype.h>
#include <string.h>

// 自定义大小写不敏感比较
int strcasecmp_custom(const char *s1, const char *s2) {
    while (*s1 && *s2) {
        int diff = tolower((unsigned char)*s1) - tolower((unsigned char)*s2);
        if (diff != 0) return diff;
        s1++;
        s2++;
    }
    return tolower((unsigned char)*s1) - tolower((unsigned char)*s2);
}

int main() {
    // "Hello" 和 "hello" 视为相等
    if (strcasecmp_custom("Hello", "hello") == 0) {
        printf("大小写不敏感: 相等\n");
    }

    return 0;
}
```

### 3.4 字符串排序

```c
#include <stdio.h>
#include <string.h>

// 冒泡排序（按字典序）
void sort_strings(char arr[][20], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (strcmp(arr[j], arr[j + 1]) > 0) {
                char temp[20];
                strcpy(temp, arr[j]);
                strcpy(arr[j], arr[j + 1]);
                strcpy(arr[j + 1], temp);
            }
        }
    }
}

int main() {
    char fruits[][20] = {"Banana", "Apple", "Cherry", "Date"};
    int n = 4;

    sort_strings(fruits, n);

    for (int i = 0; i < n; i++) {
        printf("%s\n", fruits[i]);
    }
    // Apple
    // Banana
    // Cherry
    // Date

    return 0;
}
```

## 四、常见错误

### 4.1 用 == 比较字符串

```c
// ❌ 错误：比较的是指针地址，不是字符串内容
char s1[] = "Hello";
char s2[] = "Hello";
if (s1 == s2) {     // 几乎肯定为假！
    printf("相等\n");
}

// ✅ 正确：使用 strcmp
if (strcmp(s1, s2) == 0) {
    printf("相等\n");
}
```

### 4.2 strcmp 返回值判断

```c
// ❌ 不推荐
if (strcmp(s1, s2)) {   // 返回0表示相等，0在C中为假
    printf("不相等\n");  // 逻辑反了
}

// ✅ 推荐：明确写出比较
if (strcmp(s1, s2) == 0) {
    printf("相等\n");
}
if (strcmp(s1, s2) != 0) {
    printf("不相等\n");
}
if (strcmp(s1, s2) < 0) {
    printf("s1 < s2\n");
}
```

## 五、strcmp 与 strncmp 对比

| 特性 | strcmp | strncmp |
|------|--------|---------|
| 比较范围 | 整个字符串 | 前n个字符 |
| 遇\0停止 | 是 | 是 |
| 适用场景 | 完全匹配 | 前缀匹配 |
| 返回值 | 差值或0 | 差值或0 |

## 六、重要注意事项

> **关键要点：**
> 1. **不要用 `==` 比较字符串**：`==` 比较的是指针地址
> 2. **返回值是差值**：具体值由实现决定，只应判断正负零
> 3. **区分大小写**：`strcmp("abc", "ABC")` 不相等
> 4. **空字符串比较**：`strcmp("", "") == 0` 为真
> 5. **NULL 指针**：传入 NULL 是未定义行为
> 6. **strncmp 遇到 `\0` 也会停止**：即使还没比较完 n 个字符
> 7. 对用户输入进行比较时，注意去除前后空白字符
