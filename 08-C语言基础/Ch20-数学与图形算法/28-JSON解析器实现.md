# JSON解析器实现

## 概述

JSON（JavaScript Object Notation）是轻量级的数据交换格式。本节从零实现一个简易的JSON解析器，支持所有基本类型。

---

## 1. JSON数据类型

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* JSON值类型 */
typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

/* 前向声明 */
typedef struct JsonValue JsonValue;
typedef struct JsonPair JsonPair;

/* JSON值 */
struct JsonValue {
    JsonType type;
    union {
        int bool_val;
        double number_val;
        char *string_val;
        struct {        // 数组
            JsonValue **items;
            int count;
            int capacity;
        } array;
        struct {        // 对象
            JsonPair **pairs;
            int count;
            int capacity;
        } object;
    };
};

/* 键值对 */
struct JsonPair {
    char *key;
    JsonValue *value;
};
```

---

## 2. 解析器状态

```c
/* 解析器上下文 */
typedef struct {
    const char *input;
    int pos;
    int len;
    char error[256];
} JsonParser;

/* 创建值 */
JsonValue *json_create(JsonType type) {
    JsonValue *val = (JsonValue *)calloc(1, sizeof(JsonValue));
    val->type = type;
    return val;
}

/* 跳过空白 */
static void skip_whitespace(JsonParser *p) {
    while (p->pos < p->len && isspace(p->input[p->pos])) {
        p->pos++;
    }
}

/* 查看当前字符 */
static char peek(JsonParser *p) {
    skip_whitespace(p);
    if (p->pos >= p->len) return '\0';
    return p->input[p->pos];
}

/* 消费一个字符 */
static int consume(JsonParser *p, char expected) {
    skip_whitespace(p);
    if (p->pos >= p->len || p->input[p->pos] != expected) {
        snprintf(p->error, sizeof(p->error),
                 "位置%d: 期望 '%c', 实际 '%c'", p->pos, expected,
                 p->pos < p->len ? p->input[p->pos] : '\0');
        return 0;
    }
    p->pos++;
    return 1;
}
```

---

## 3. 解析各种类型

### 3.1 解析字符串

```c
/* 解析字符串 */
JsonValue *parse_string(JsonParser *p) {
    if (!consume(p, '"')) return NULL;

    int start = p->pos;
    int capacity = 64;
    char *str = (char *)malloc(capacity);
    int len = 0;

    while (p->pos < p->len && p->input[p->pos] != '"') {
        char c = p->input[p->pos];

        if (c == '\\') {
            p->pos++;
            if (p->pos >= p->len) {
                free(str);
                snprintf(p->error, sizeof(p->error), "意外的字符串结束");
                return NULL;
            }
            switch (p->input[p->pos]) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                default:
                    free(str);
                    snprintf(p->error, sizeof(p->error), "未知转义字符");
                    return NULL;
            }
        }

        if (len >= capacity - 1) {
            capacity *= 2;
            str = (char *)realloc(str, capacity);
        }
        str[len++] = c;
        p->pos++;
    }

    if (!consume(p, '"')) {
        free(str);
        return NULL;
    }

    str[len] = '\0';

    JsonValue *val = json_create(JSON_STRING);
    val->string_val = str;
    return val;
}
```

### 3.2 解析数字

```c
/* 解析数字 */
JsonValue *parse_number(JsonParser *p) {
    skip_whitespace(p);
    int start = p->pos;

    // 处理负号
    if (p->pos < p->len && p->input[p->pos] == '-') p->pos++;

    // 处理整数部分
    if (p->pos >= p->len || !isdigit(p->input[p->pos])) {
        snprintf(p->error, sizeof(p->error), "无效的数字");
        return NULL;
    }
    while (p->pos < p->len && isdigit(p->input[p->pos])) p->pos++;

    // 处理小数部分
    if (p->pos < p->len && p->input[p->pos] == '.') {
        p->pos++;
        while (p->pos < p->len && isdigit(p->input[p->pos])) p->pos++;
    }

    // 处理指数部分
    if (p->pos < p->len && (p->input[p->pos] == 'e' || p->input[p->pos] == 'E')) {
        p->pos++;
        if (p->pos < p->len && (p->input[p->pos] == '+' || p->input[p->pos] == '-'))
            p->pos++;
        while (p->pos < p->len && isdigit(p->input[p->pos])) p->pos++;
    }

    int len = p->pos - start;
    char *num_str = (char *)malloc(len + 1);
    memcpy(num_str, p->input + start, len);
    num_str[len] = '\0';

    JsonValue *val = json_create(JSON_NUMBER);
    val->number_val = atof(num_str);
    free(num_str);

    return val;
}
```

### 3.3 解析字面值

```c
/* 解析 true/false/null */
JsonValue *parse_literal(JsonParser *p) {
    skip_whitespace(p);

    if (strncmp(p->input + p->pos, "true", 4) == 0) {
        p->pos += 4;
        JsonValue *val = json_create(JSON_BOOL);
        val->bool_val = 1;
        return val;
    }

    if (strncmp(p->input + p->pos, "false", 5) == 0) {
        p->pos += 5;
        JsonValue *val = json_create(JSON_BOOL);
        val->bool_val = 0;
        return val;
    }

    if (strncmp(p->input + p->pos, "null", 4) == 0) {
        p->pos += 4;
        return json_create(JSON_NULL);
    }

    snprintf(p->error, sizeof(p->error), "未知的字面值");
    return NULL;
}
```

### 3.4 解析数组

```c
/* 前向声明 */
JsonValue *parse_value(JsonParser *p);

JsonValue *parse_array(JsonParser *p) {
    if (!consume(p, '[')) return NULL;

    JsonValue *val = json_create(JSON_ARRAY);
    val->array.capacity = 8;
    val->array.items = (JsonValue **)malloc(val->array.capacity * sizeof(JsonValue *));
    val->array.count = 0;

    if (peek(p) == ']') {
        p->pos++;
        return val;
    }

    while (1) {
        JsonValue *item = parse_value(p);
        if (!item) return NULL;

        if (val->array.count >= val->array.capacity) {
            val->array.capacity *= 2;
            val->array.items = (JsonValue **)realloc(
                val->array.items, val->array.capacity * sizeof(JsonValue *));
        }
        val->array.items[val->array.count++] = item;

        if (peek(p) == ',') {
            p->pos++;
        } else {
            break;
        }
    }

    if (!consume(p, ']')) return NULL;
    return val;
}
```

### 3.5 解析对象

```c
JsonValue *parse_object(JsonParser *p) {
    if (!consume(p, '{')) return NULL;

    JsonValue *val = json_create(JSON_OBJECT);
    val->object.capacity = 8;
    val->object.pairs = (JsonPair **)malloc(val->object.capacity * sizeof(JsonPair *));
    val->object.count = 0;

    if (peek(p) == '}') {
        p->pos++;
        return val;
    }

    while (1) {
        // 解析键
        JsonValue *key_val = parse_string(p);
        if (!key_val) return NULL;

        if (!consume(p, ':')) return NULL;

        // 解析值
        JsonValue *value = parse_value(p);
        if (!value) return NULL;

        // 创建键值对
        JsonPair *pair = (JsonPair *)malloc(sizeof(JsonPair));
        pair->key = key_val->string_val;
        pair->value = value;
        free(key_val);  // 只释放外壳，保留string_val

        if (val->object.count >= val->object.capacity) {
            val->object.capacity *= 2;
            val->object.pairs = (JsonPair **)realloc(
                val->object.pairs, val->object.capacity * sizeof(JsonPair *));
        }
        val->object.pairs[val->object.count++] = pair;

        if (peek(p) == ',') {
            p->pos++;
        } else {
            break;
        }
    }

    if (!consume(p, '}')) return NULL;
    return val;
}
```

### 3.6 统一解析入口

```c
JsonValue *parse_value(JsonParser *p) {
    skip_whitespace(p);
    if (p->pos >= p->len) return NULL;

    char c = p->input[p->pos];
    if (c == '"')  return parse_string(p);
    if (c == '[')  return parse_array(p);
    if (c == '{')  return parse_object(p);
    if (c == 't' || c == 'f' || c == 'n') return parse_literal(p);
    if (c == '-' || isdigit(c)) return parse_number(p);

    snprintf(p->error, sizeof(p->error), "位置%d: 未知字符 '%c'", p->pos, c);
    return NULL;
}

/*
 * 解析JSON字符串
 * 返回JSON值的根节点，失败返回NULL
 */
JsonValue *json_parse(const char *input, char *error_buf, int error_size) {
    JsonParser p;
    p.input = input;
    p.pos = 0;
    p.len = (int)strlen(input);
    p.error[0] = '\0';

    JsonValue *root = parse_value(&p);

    if (!root && error_buf) {
        strncpy(error_buf, p.error, error_size - 1);
        error_buf[error_size - 1] = '\0';
    }

    return root;
}
```

---

## 4. 访问接口

```c
/* 获取数组长度 */
int json_array_size(JsonValue *val) {
    return (val && val->type == JSON_ARRAY) ? val->array.count : 0;
}

/* 获取数组元素 */
JsonValue *json_array_get(JsonValue *val, int index) {
    if (!val || val->type != JSON_ARRAY || index < 0 || index >= val->array.count)
        return NULL;
    return val->array.items[index];
}

/* 获取对象键值 */
JsonValue *json_object_get(JsonValue *val, const char *key) {
    if (!val || val->type != JSON_OBJECT) return NULL;
    for (int i = 0; i < val->object.count; i++) {
        if (strcmp(val->object.pairs[i]->key, key) == 0) {
            return val->object.pairs[i]->value;
        }
    }
    return NULL;
}

/* 获取对象键的数量 */
int json_object_size(JsonValue *val) {
    return (val && val->type == JSON_OBJECT) ? val->object.count : 0;
}
```

---

## 5. 打印与释放

```c
/* 打印JSON（美化输出） */
void json_print(JsonValue *val, int indent) {
    if (!val) { printf("null"); return; }

    const char *pad = "    ";

    switch (val->type) {
        case JSON_NULL:   printf("null"); break;
        case JSON_BOOL:   printf(val->bool_val ? "true" : "false"); break;
        case JSON_NUMBER:
            if (val->number_val == (long long)val->number_val)
                printf("%.0f", val->number_val);
            else
                printf("%g", val->number_val);
            break;
        case JSON_STRING: printf("\"%s\"", val->string_val); break;
        case JSON_ARRAY:
            printf("[\n");
            for (int i = 0; i < val->array.count; i++) {
                for (int j = 0; j <= indent; j++) printf("%s", pad);
                json_print(val->array.items[i], indent + 1);
                if (i < val->array.count - 1) printf(",");
                printf("\n");
            }
            for (int j = 0; j < indent; j++) printf("%s", pad);
            printf("]");
            break;
        case JSON_OBJECT:
            printf("{\n");
            for (int i = 0; i < val->object.count; i++) {
                for (int j = 0; j <= indent; j++) printf("%s", pad);
                printf("\"%s\": ", val->object.pairs[i]->key);
                json_print(val->object.pairs[i]->value, indent + 1);
                if (i < val->object.count - 1) printf(",");
                printf("\n");
            }
            for (int j = 0; j < indent; j++) printf("%s", pad);
            printf("}");
            break;
    }
}

/* 释放JSON */
void json_free(JsonValue *val) {
    if (!val) return;

    switch (val->type) {
        case JSON_STRING: free(val->string_val); break;
        case JSON_ARRAY:
            for (int i = 0; i < val->array.count; i++)
                json_free(val->array.items[i]);
            free(val->array.items);
            break;
        case JSON_OBJECT:
            for (int i = 0; i < val->object.count; i++) {
                free(val->object.pairs[i]->key);
                json_free(val->object.pairs[i]->value);
                free(val->object.pairs[i]);
            }
            free(val->object.pairs);
            break;
        default: break;
    }
    free(val);
}
```

---

## 6. 测试

```c
int main() {
    const char *json_str =
        "{"
        "  \"name\": \"张三\","
        "  \"age\": 25,"
        "  \"scores\": [90, 85, 92],"
        "  \"address\": {"
        "    \"city\": \"北京\","
        "    \"zip\": \"100000\""
        "  },"
        "  \"active\": true,"
        "  \"notes\": null"
        "}";

    char error[256];
    JsonValue *root = json_parse(json_str, error, sizeof(error));

    if (!root) {
        printf("解析失败: %s\n", error);
        return 1;
    }

    printf("=== 解析结果 ===\n\n");
    json_print(root, 0);
    printf("\n\n");

    // 访问数据
    JsonValue *name = json_object_get(root, "name");
    if (name) printf("姓名: %s\n", name->string_val);

    JsonValue *age = json_object_get(root, "age");
    if (age) printf("年龄: %.0f\n", age->number_val);

    JsonValue *scores = json_object_get(root, "scores");
    if (scores) {
        printf("成绩: ");
        for (int i = 0; i < json_array_size(scores); i++) {
            printf("%.0f ", json_array_get(scores, i)->number_val);
        }
        printf("\n");
    }

    JsonValue *city = json_object_get(
        json_object_get(root, "address"), "city");
    if (city) printf("城市: %s\n", city->string_val);

    json_free(root);
    return 0;
}
```

---

## 小结

- JSON有6种基本类型：null、bool、number、string、array、object
- 递归下降解析是实现解析器的经典方法
- 字符串解析需处理转义字符（`\n`, `\"`, `\\`等）
- 解析器需要良好的错误处理，报告位置和错误原因
- 实际工程中可使用成熟库如cJSON、Jansson
