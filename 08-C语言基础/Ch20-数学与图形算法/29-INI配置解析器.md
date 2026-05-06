# INI配置解析器

## 概述

INI是一种简单的配置文件格式，广泛用于Windows应用程序。本节实现一个完整的INI文件读写解析器。

---

## 1. INI格式说明

```ini
; 这是注释
[database]
host = localhost
port = 3306
name = mydb

[server]
host = 0.0.0.0
port = 8080
debug = true
```

---

## 2. 数据结构

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LEN 1024
#define MAX_KEY_LEN  256
#define MAX_VAL_LEN  768

/* 键值对 */
typedef struct INIEntry {
    char key[MAX_KEY_LEN];
    char value[MAX_VAL_LEN];
    struct INIEntry *next;
} INIEntry;

/* 配置段 */
typedef struct INISection {
    char name[MAX_KEY_LEN];
    INIEntry *entries;
    struct INISection *next;
} INISection;

/* INI配置文件 */
typedef struct {
    INISection *sections;     // 段链表
    INISection *last_section; // 尾指针，加速追加
} INIConfig;
```

---

## 3. 创建与销毁

```c
/* 创建空配置 */
INIConfig *ini_create() {
    INIConfig *ini = (INIConfig *)calloc(1, sizeof(INIConfig));
    return ini;
}

/* 释放所有内存 */
void ini_free(INIConfig *ini) {
    if (!ini) return;

    INISection *sec = ini->sections;
    while (sec) {
        INIEntry *entry = sec->entries;
        while (entry) {
            INIEntry *next_entry = entry->next;
            free(entry);
            entry = next_entry;
        }
        INISection *next_sec = sec->next;
        free(sec);
        sec = next_sec;
    }
    free(ini);
}
```

---

## 4. 解析

### 4.1 辅助函数

```c
/* 去除首尾空白 */
static char *strip(char *str) {
    // 去除尾部空白
    int len = (int)strlen(str);
    while (len > 0 && isspace(str[len - 1])) str[--len] = '\0';
    // 跳过前导空白
    while (*str && isspace(*str)) str++;
    return str;
}

/* 检查是否为注释行 */
static int is_comment(const char *line) {
    return (*line == ';' || *line == '#');
}

/* 查找或创建段 */
static INISection *find_or_create_section(INIConfig *ini, const char *name) {
    INISection *sec = ini->sections;
    while (sec) {
        if (strcmp(sec->name, name) == 0) return sec;
        sec = sec->next;
    }

    // 创建新段
    sec = (INISection *)calloc(1, sizeof(INISection));
    strncpy(sec->name, name, MAX_KEY_LEN - 1);

    if (ini->last_section) {
        ini->last_section->next = sec;
    } else {
        ini->sections = sec;
    }
    ini->last_section = sec;

    return sec;
}
```

### 4.2 解析文件

```c
/* 解析INI文件 */
INIConfig *ini_parse_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;

    INIConfig *ini = ini_create();
    INISection *current_section = NULL;
    char line[MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp)) {
        char *trimmed = strip(line);

        // 跳过空行和注释
        if (*trimmed == '\0' || is_comment(trimmed)) continue;

        // 解析段名 [section]
        if (*trimmed == '[') {
            char *end = strchr(trimmed, ']');
            if (end) {
                *end = '\0';
                char *sec_name = strip(trimmed + 1);
                current_section = find_or_create_section(ini, sec_name);
            }
            continue;
        }

        // 解析键值对 key = value
        char *eq = strchr(trimmed, '=');
        if (eq && current_section) {
            *eq = '\0';
            char *key = strip(trimmed);
            char *value = strip(eq + 1);

            // 去除值的引号
            int vlen = (int)strlen(value);
            if (vlen >= 2 && ((value[0] == '"' && value[vlen-1] == '"') ||
                              (value[0] == '\'' && value[vlen-1] == '\''))) {
                value[vlen-1] = '\0';
                value++;
            }

            INIEntry *entry = (INIEntry *)calloc(1, sizeof(INIEntry));
            strncpy(entry->key, key, MAX_KEY_LEN - 1);
            strncpy(entry->value, value, MAX_VAL_LEN - 1);

            // 添加到段的链表尾部
            if (!current_section->entries) {
                current_section->entries = entry;
            } else {
                INIEntry *last = current_section->entries;
                while (last->next) last = last->next;
                last->next = entry;
            }
        }
    }

    fclose(fp);
    return ini;
}

/* 从字符串解析 */
INIConfig *ini_parse_string(const char *str) {
    INIConfig *ini = ini_create();
    INISection *current_section = NULL;

    char *copy = strdup(str);
    char *line = strtok(copy, "\n");

    while (line) {
        char *trimmed = strip(line);

        if (*trimmed == '\0' || is_comment(trimmed)) {
            line = strtok(NULL, "\n");
            continue;
        }

        if (*trimmed == '[') {
            char *end = strchr(trimmed, ']');
            if (end) {
                *end = '\0';
                current_section = find_or_create_section(ini, trimmed + 1);
            }
            line = strtok(NULL, "\n");
            continue;
        }

        char *eq = strchr(trimmed, '=');
        if (eq && current_section) {
            *eq = '\0';
            char *key = strip(trimmed);
            char *value = strip(eq + 1);

            INIEntry *entry = (INIEntry *)calloc(1, sizeof(INIEntry));
            strncpy(entry->key, key, MAX_KEY_LEN - 1);
            strncpy(entry->value, value, MAX_VAL_LEN - 1);

            if (!current_section->entries) {
                current_section->entries = entry;
            } else {
                INIEntry *last = current_section->entries;
                while (last->next) last = last->next;
                last->next = entry;
            }
        }

        line = strtok(NULL, "\n");
    }

    free(copy);
    return ini;
}
```

---

## 5. 读取值

```c
/* 获取字符串值 */
const char *ini_get_string(INIConfig *ini, const char *section,
                           const char *key, const char *default_val) {
    if (!ini) return default_val;

    INISection *sec = ini->sections;
    while (sec) {
        if (strcmp(sec->name, section) == 0) {
            INIEntry *entry = sec->entries;
            while (entry) {
                if (strcmp(entry->key, key) == 0) {
                    return entry->value;
                }
                entry = entry->next;
            }
            return default_val;
        }
        sec = sec->next;
    }
    return default_val;
}

/* 获取整数值 */
int ini_get_int(INIConfig *ini, const char *section,
                const char *key, int default_val) {
    const char *val = ini_get_string(ini, section, key, NULL);
    return val ? atoi(val) : default_val;
}

/* 获取浮点值 */
double ini_get_double(INIConfig *ini, const char *section,
                      const char *key, double default_val) {
    const char *val = ini_get_string(ini, section, key, NULL);
    return val ? atof(val) : default_val;
}

/* 获取布尔值 */
int ini_get_bool(INIConfig *ini, const char *section,
                 const char *key, int default_val) {
    const char *val = ini_get_string(ini, section, key, NULL);
    if (!val) return default_val;

    if (strcasecmp(val, "true") == 0 || strcmp(val, "1") == 0 ||
        strcasecmp(val, "yes") == 0) return 1;
    if (strcasecmp(val, "false") == 0 || strcmp(val, "0") == 0 ||
        strcasecmp(val, "no") == 0) return 0;

    return default_val;
}
```

---

## 6. 写入值

```c
/* 设置键值对 */
void ini_set(INIConfig *ini, const char *section,
             const char *key, const char *value) {
    INISection *sec = find_or_create_section(ini, section);

    // 查找现有键
    INIEntry *entry = sec->entries;
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            strncpy(entry->value, value, MAX_VAL_LEN - 1);
            return;
        }
        entry = entry->next;
    }

    // 创建新键值对
    entry = (INIEntry *)calloc(1, sizeof(INIEntry));
    strncpy(entry->key, key, MAX_KEY_LEN - 1);
    strncpy(entry->value, value, MAX_VAL_LEN - 1);

    if (!sec->entries) {
        sec->entries = entry;
    } else {
        INIEntry *last = sec->entries;
        while (last->next) last = last->next;
        last->next = entry;
    }
}

/* 删除键 */
void ini_delete_key(INIConfig *ini, const char *section, const char *key) {
    INISection *sec = ini->sections;
    while (sec) {
        if (strcmp(sec->name, section) == 0) {
            INIEntry **prev = &sec->entries;
            INIEntry *entry = sec->entries;
            while (entry) {
                if (strcmp(entry->key, key) == 0) {
                    *prev = entry->next;
                    free(entry);
                    return;
                }
                prev = &entry->next;
                entry = entry->next;
            }
            return;
        }
        sec = sec->next;
    }
}
```

---

## 7. 写入文件

```c
/* 保存INI文件 */
int ini_save(INIConfig *ini, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return -1;

    INISection *sec = ini->sections;
    while (sec) {
        fprintf(fp, "[%s]\n", sec->name);

        INIEntry *entry = sec->entries;
        while (entry) {
            fprintf(fp, "%s = %s\n", entry->key, entry->value);
            entry = entry->next;
        }

        fprintf(fp, "\n");
        sec = sec->next;
    }

    fclose(fp);
    return 0;
}
```

---

## 8. 遍历

```c
/* 遍历所有段 */
void ini_foreach_section(INIConfig *ini,
                         void (*callback)(const char *section, void *data),
                         void *data) {
    INISection *sec = ini->sections;
    while (sec) {
        callback(sec->name, data);
        sec = sec->next;
    }
}

/* 遍历段中的所有键值对 */
void ini_foreach_entry(INIConfig *ini, const char *section,
                       void (*callback)(const char *key, const char *value,
                                        void *data),
                       void *data) {
    INISection *sec = ini->sections;
    while (sec) {
        if (strcmp(sec->name, section) == 0) {
            INIEntry *entry = sec->entries;
            while (entry) {
                callback(entry->key, entry->value, data);
                entry = entry->next;
            }
            return;
        }
        sec = sec->next;
    }
}
```

---

## 9. 测试

```c
void print_section(const char *name, void *data) {
    printf("  [%s]\n", name);
}

void print_entry(const char *key, const char *value, void *data) {
    printf("    %s = %s\n", key, value);
}

int main() {
    // 从字符串解析
    const char *ini_text =
        "; 配置文件示例\n"
        "[database]\n"
        "host = localhost\n"
        "port = 3306\n"
        "name = mydb\n"
        "\n"
        "[server]\n"
        "host = 0.0.0.0\n"
        "port = 8080\n"
        "debug = true\n";

    INIConfig *ini = ini_parse_string(ini_text);

    // 读取值
    printf("数据库主机: %s\n", ini_get_string(ini, "database", "host", "N/A"));
    printf("数据库端口: %d\n", ini_get_int(ini, "database", "port", 0));
    printf("服务器调试: %s\n", ini_get_bool(ini, "server", "debug", 0) ? "是" : "否");
    printf("未知配置: %s\n", ini_get_string(ini, "server", "timeout", "30"));

    // 设置值
    ini_set(ini, "server", "timeout", "60");
    ini_set(ini, "logging", "level", "info");

    // 遍历
    printf("\n所有段:\n");
    ini_foreach_section(ini, print_section, NULL);

    printf("\n数据库段的所有配置:\n");
    ini_foreach_entry(ini, "database", print_entry, NULL);

    ini_free(ini);

    return 0;
}
```

---

## 小结

- INI格式简单，适合少量配置项的场景
- 解析时需注意去除空白、处理注释、处理引号
- 使用链表存储段和键值对，方便动态增删
- 也可以用哈希表加速键的查找
