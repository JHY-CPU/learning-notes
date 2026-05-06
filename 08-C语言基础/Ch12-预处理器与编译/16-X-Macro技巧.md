# 16 - X-Macro 技巧

## X-Macro 模式概述

X-Macro 是一种利用预处理器实现**代码生成**的设计模式。其核心思想是：将一个列表定义为宏，然后多次引用该宏，每次用不同的展开方式来生成不同的代码。

这种技术可以消除"枚举与字符串表不同步"这类常见问题。

## 基本原理

```c
// 步骤1: 定义列表（X-List）
#define COLOR_LIST \
    X(RED) \
    X(GREEN) \
    X(BLUE) \
    X(YELLOW)

// 步骤2: 定义不同的 X 宏，生成不同的代码
// 第一次：生成枚举
#define X(name) COLOR_##name,
enum Color {
    COLOR_LIST
    COLOR_COUNT
};
#undef X

// 第二次：生成字符串表
#define X(name) #name,
const char *color_names[] = {
    COLOR_LIST
};
#undef X
```

展开后的结果：

```c
// 枚举展开为：
enum Color {
    COLOR_RED,
    COLOR_GREEN,
    COLOR_BLUE,
    COLOR_YELLOW,
    COLOR_COUNT
};

// 字符串表展开为：
const char *color_names[] = {
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
};
```

## 经典应用：错误码系统

```c
// error_codes.h
#ifndef ERROR_CODES_H
#define ERROR_CODES_H

// 定义所有错误码
#define ERROR_LIST \
    X(ERR_OK,           "成功") \
    X(ERR_NULL_PTR,     "空指针错误") \
    X(ERR_OUT_OF_MEM,   "内存不足") \
    X(ERR_INVALID_ARG,  "无效参数") \
    X(ERR_FILE_NOT_FOUND, "文件未找到") \
    X(ERR_PERMISSION,   "权限不足") \
    X(ERR_TIMEOUT,      "操作超时") \
    X(ERR_UNKNOWN,      "未知错误")

// 生成枚举
typedef enum {
#define X(code, msg) code,
    ERROR_LIST
#undef X
    ERR_COUNT
} ErrorCode;

// 生成错误消息表
static const char *error_messages[] = {
#define X(code, msg) [code] = msg,
    ERROR_LIST
#undef X
};

// 获取错误消息的函数
static inline const char *error_to_string(ErrorCode code) {
    if (code >= 0 && code < ERR_COUNT)
        return error_messages[code];
    return "无效错误码";
}

#endif /* ERROR_CODES_H */
```

使用：

```c
#include "error_codes.h"

ErrorCode result = some_operation();
if (result != ERR_OK) {
    printf("错误: %s (代码: %d)\n", error_to_string(result), result);
}
```

## 应用：状态机

```c
// state_machine.h
#define STATE_LIST \
    X(STATE_IDLE,     "空闲") \
    X(STATE_RUNNING,  "运行中") \
    X(STATE_PAUSED,   "已暂停") \
    X(STATE_STOPPED,  "已停止") \
    X(STATE_ERROR,    "错误状态")

// 生成枚举
typedef enum {
#define X(state, desc) state,
    STATE_LIST
#undef X
    STATE_COUNT
} State;

// 生成描述表
static const char *state_descriptions[] = {
#define X(state, desc) [state] = desc,
    STATE_LIST
#undef X
};

// 事件列表
#define EVENT_LIST \
    X(EVT_START,  "启动") \
    X(EVT_PAUSE,  "暂停") \
    X(EVT_RESUME, "恢复") \
    X(EVT_STOP,   "停止") \
    X(EVT_ERROR,  "出错")

typedef enum {
#define X(evt, desc) evt,
    EVENT_LIST
#undef X
    EVT_COUNT
} Event;
```

## 应用：命令行参数解析

```c
// commands.h
#define COMMAND_LIST \
    X(CMD_HELP,    "help",    "显示帮助信息",  0) \
    X(CMD_VERSION, "version", "显示版本信息",  0) \
    X(CMD_RUN,     "run",     "运行程序",      1) \
    X(CMD_CONFIG,  "config",  "配置参数",      2) \
    X(CMD_DEBUG,   "debug",   "调试模式",      1)

// 生成枚举
typedef enum {
#define X(id, name, desc, args) id,
    COMMAND_LIST
#undef X
    CMD_COUNT
} CommandId;

// 命令信息结构体
typedef struct {
    CommandId id;
    const char *name;
    const char *description;
    int num_args;
} CommandInfo;

// 生成命令表
static const CommandInfo commands[] = {
#define X(id, name, desc, args) { id, name, desc, args },
    COMMAND_LIST
#undef X
};

// 查找命令
static inline const CommandInfo *find_command(const char *name) {
    for (int i = 0; i < CMD_COUNT; i++) {
        if (strcmp(commands[i].name, name) == 0)
            return &commands[i];
    }
    return NULL;
}
```

## 应用：网络协议消息定义

```c
// protocol.h
#define MESSAGE_TYPES \
    X(MSG_CONNECT,    0x01, "连接请求") \
    X(MSG_DISCONNECT, 0x02, "断开连接") \
    X(MSG_DATA,       0x03, "数据传输") \
    X(MSG_ACK,        0x04, "确认应答") \
    X(MSG_NACK,       0x05, "否定应答") \
    X(MSG_HEARTBEAT,  0x06, "心跳包")

// 生成枚举
typedef enum {
#define X(name, code, desc) name = code,
    MESSAGE_TYPES
#undef X
} MessageType;

// 生成描述函数
static inline const char *msg_type_to_string(MessageType type) {
    switch (type) {
#define X(name, code, desc) case code: return desc;
        MESSAGE_TYPES
#undef X
        default: return "未知消息类型";
    }
}

// 生成消息名称函数
static inline const char *msg_type_to_name(MessageType type) {
    switch (type) {
#define X(name, code, desc) case code: return #name;
        MESSAGE_TYPES
#undef X
        default: return "UNKNOWN";
    }
}
```

## 带参数的 X-Macro

```c
// 定义一个更灵活的 X-Macro，支持不同类型的数据
#define DATA_TYPES \
    X(int,    INT,    "整数",   "%d") \
    X(float,  FLOAT,  "浮点数", "%.2f") \
    X(double, DOUBLE, "双精度", "%.4f") \
    X(char*,  STRING, "字符串", "%s")

// 生成类型枚举
typedef enum {
#define X(c_type, enum_name, desc, fmt) TYPE_##enum_name,
    DATA_TYPES
#undef X
    TYPE_COUNT
} DataType;

// 生成打印函数
#define X(c_type, enum_name, desc, fmt) \
    static void print_##enum_name(c_type value) { \
        printf("[" desc "] " fmt "\n", value); \
    }
DATA_TYPES
#undef X

// 使用
print_INT(42);
print_FLOAT(3.14f);
print_STRING("hello");
```

## X-Macro 的优缺点

### 优点

1. **单一数据源**：修改列表一处，所有生成的代码自动同步
2. **减少重复**：避免手动维护枚举和字符串表的一致性
3. **易于扩展**：添加新条目只需修改列表宏
4. **编译时生成**：没有运行时开销

### 缺点

1. **可读性**：对不熟悉此模式的开发者来说较难理解
2. **调试困难**：宏展开后的代码难以调试
3. **IDE支持**：某些 IDE 可能无法正确索引生成的代码
4. **错误消息**：编译错误可能指向宏展开后的代码，难以定位

## 重要注意事项

> **关键点总结**：
> 1. X-Macro 的核心是**一个列表定义，多次引用**，每次用不同的 `X` 展开
> 2. 最常见的应用是**枚举与字符串表同步**
> 3. 列表中的每个条目可以携带任意数量的数据（错误消息、代码、描述等）
> 4. 每次引用前要 `#undef X`，引用后要 `#undef X`
> 5. X-Macro 适合**数据驱动的代码生成**，不适合复杂的逻辑生成
> 6. 当枚举需要与字符串、ID、描述等信息同步时，X-Macro 是最佳选择
