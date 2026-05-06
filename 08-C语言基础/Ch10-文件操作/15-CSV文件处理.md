# 15 - CSV文件处理

## 一、CSV格式概述

### CSV基本规则

CSV（Comma-Separated Values）是最常见的表格数据交换格式：

```
姓名,年龄,城市,薪资
张三,25,北京,8500.50
李四,30,上海,12000.00
"王五",28,"广州,深圳",9500.75
```

### CSV格式要点

| 规则 | 示例 |
|------|------|
| 每行一条记录 | 换行符分隔 |
| 字段间用逗号分隔 | `A,B,C` |
| 含逗号/换行的字段需引号包围 | `"广州,深圳"` |
| 字段内引号用双引号转义 | `"他说""你好"""` |
| 第一行通常是标题（惯例） | `姓名,年龄,城市` |

## 二、写入CSV文件

### 基本写入

```c
#include <stdio.h>
#include <string.h>

// 检查字段是否需要引号包围
int needs_quoting(const char *field) {
    return strchr(field, ',') != NULL ||
           strchr(field, '"') != NULL ||
           strchr(field, '\n') != NULL;
}

// 写入一个字段
void write_csv_field(FILE *fp, const char *field) {
    if (!needs_quoting(field)) {
        fputs(field, fp);
        return;
    }

    // 需要引号包围，内部引号加倍
    fputc('"', fp);
    for (const char *p = field; *p; p++) {
        if (*p == '"') fputc('"', fp);  // 引号转义
        fputc(*p, fp);
    }
    fputc('"', fp);
}

// 写入一行
void write_csv_row(FILE *fp, const char *fields[], int nfields) {
    for (int i = 0; i < nfields; i++) {
        if (i > 0) fputc(',', fp);
        write_csv_field(fp, fields[i]);
    }
    fputc('\n', fp);
}

int main() {
    FILE *fp = fopen("data.csv", "w");
    if (!fp) { perror("创建失败"); return 1; }

    // 写入BOM（UTF-8标识，Excel兼容）
    // fputs("\xEF\xBB\xBF", fp);  // 可选

    // 写入标题行
    const char *header[] = {"姓名", "年龄", "城市", "薪资"};
    write_csv_row(fp, header, 4);

    // 写入数据行
    const char *row1[] = {"张三", "25", "北京", "8500.50"};
    const char *row2[] = {"李四", "30", "上海,浦东", "12000.00"};
    const char *row3[] = {"王\"五\"", "28", "广州", "9500.75"};

    write_csv_row(fp, row1, 4);
    write_csv_row(fp, row2, 4);
    write_csv_row(fp, row3, 4);

    fclose(fp);
    printf("CSV写入完成\n");
    return 0;
}
```

## 三、读取CSV文件

### CSV解析器

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FIELDS 64
#define MAX_FIELD_LEN 1024

// 解析一行CSV，返回字段数
int parse_csv_line(char *line, char fields[][MAX_FIELD_LEN], int max_fields) {
    int nfields = 0;
    char *p = line;

    while (*p && nfields < max_fields) {
        char *field_start = fields[nfields];

        if (*p == '"') {
            // 引号包围的字段
            p++;  // 跳过开头引号
            while (*p) {
                if (*p == '"') {
                    p++;
                    if (*p == '"') {
                        // 双引号转义
                        *field_start++ = '"';
                        p++;
                    } else {
                        // 字段结束
                        break;
                    }
                } else {
                    *field_start++ = *p++;
                }
            }
        } else {
            // 无引号字段
            while (*p && *p != ',' && *p != '\n' && *p != '\r') {
                *field_start++ = *p++;
            }
        }

        *field_start = '\0';
        nfields++;

        // 跳过逗号
        if (*p == ',') p++;
    }

    return nfields;
}

int main() {
    FILE *fp = fopen("data.csv", "r");
    if (!fp) { perror("打开失败"); return 1; }

    char line[4096];
    char fields[MAX_FIELDS][MAX_FIELD_LEN];
    int row = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        // 跳过空行
        if (line[0] == '\n' || line[0] == '\r') continue;

        // 去除行末换行符
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }

        int nfields = parse_csv_line(line, fields, MAX_FIELDS);

        printf("第%d行 (%d个字段):", row, nfields);
        for (int i = 0; i < nfields; i++) {
            printf(" [%s]", fields[i]);
        }
        printf("\n");

        row++;
    }

    fclose(fp);
    return 0;
}
```

## 四、结构化CSV读写

### 完整的学生CSV系统

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int    id;
    char   name[50];
    int    age;
    char   city[30];
    double score;
} Student;

void student_to_csv(FILE *fp, const Student *s) {
    fprintf(fp, "%d,%s,%d,%s,%.1f\n",
            s->id, s->name, s->age, s->city, s->score);
}

int csv_to_student(const char *line, Student *s) {
    return sscanf(line, "%d,%49[^,],%d,%29[^,],%lf",
                  &s->id, s->name, &s->age, s->city, &s->score) == 5;
}

void save_students(const char *filename, Student students[], int count) {
    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("保存失败"); return; }

    fprintf(fp, "学号,姓名,年龄,城市,成绩\n");
    for (int i = 0; i < count; i++) {
        student_to_csv(fp, &students[i]);
    }

    fclose(fp);
}

int load_students(const char *filename, Student students[], int max_count) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("加载失败"); return 0; }

    char line[256];
    int count = 0;

    // 跳过标题行
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp) && count < max_count) {
        if (csv_to_student(line, &students[count])) {
            count++;
        }
    }

    fclose(fp);
    return count;
}

void print_students(Student students[], int count) {
    printf("%-6s %-10s %-6s %-10s %s\n",
           "学号", "姓名", "年龄", "城市", "成绩");
    printf("------------------------------------------\n");
    for (int i = 0; i < count; i++) {
        printf("%-6d %-10s %-6d %-10s %.1f\n",
               students[i].id, students[i].name,
               students[i].age, students[i].city,
               students[i].score);
    }
}

int main() {
    Student students[] = {
        {1001, "张三", 20, "北京", 85.5},
        {1002, "李四", 21, "上海", 92.0},
        {1003, "王五", 19, "广州", 78.3},
        {1004, "赵六", 22, "深圳", 88.7}
    };

    // 保存到CSV
    save_students("students.csv", students, 4);
    printf("数据已保存到 students.csv\n\n");

    // 从CSV加载
    Student loaded[100];
    int count = load_students("students.csv", loaded, 100);

    printf("从CSV加载了 %d 条记录:\n", count);
    print_students(loaded, count);

    return 0;
}
```

## 五、重点与注意事项

> **要点总结：**
>
> 1. **含逗号、引号、换行的字段必须用双引号包围**
> 2. **字段内的引号用两个连续双引号表示**：`"` -> `""`
> 3. 不同系统换行符不同（`\r\n` vs `\n`），解析时需兼容处理
> 4. `sscanf` 解析CSV不够灵活，复杂场景应使用手动解析
> 5. UTF-8 CSV文件带BOM时，Excel能正确识别中文字符
> 6. 生产环境建议使用成熟的CSV库（如 `libcsv`）
