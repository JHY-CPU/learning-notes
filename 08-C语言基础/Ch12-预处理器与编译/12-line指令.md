# 12 - #line 指令

## #line 指令概述

`#line` 指令用于改变预处理器中的**行号**和**文件名**信息。编译器使用这些信息生成错误消息和调试信息，`#line` 允许程序员手动修改这些值。

## 基本语法

```c
// 形式1：只修改行号
#line 行号

// 形式2：修改行号和文件名
#line 行号 "文件名"
```

修改后，后续代码的行号从指定值开始递增，文件名变为指定的名称。

## 基本示例

```c
#include <stdio.h>

int main(void) {
    printf("这是第 %d 行\n", __LINE__);  // 输出实际行号
    
#line 100 "custom_file.c"
    
    printf("文件: %s, 行号: %d\n", __FILE__, __LINE__);
    // 输出: 文件: custom_file.c, 行号: 100
    
    printf("下一行: %d\n", __LINE__);
    // 输出: 下一行: 101
    
    return 0;
}
```

## 错误报告中的应用

### 代码生成器

`#line` 最常见的用途是代码生成器。生成的代码希望错误消息指向**原始源文件**而非生成的中间文件：

```
// mycode.dsl（原始 DSL 文件）
第1行: function main() {
第2行:   print("Hello")
第3行: }

// → 代码生成器输出 →
```

```c
// generated.c（自动生成的文件）
#line 1 "mycode.dsl"
void main() {
    #line 2 "mycode.dsl"
    printf("Hello");
    #line 3 "mycode.dsl"
}
```

如果第2行有错误，编译器会报告：
```
mycode.dsl:2: error: ...
```
而不是：
```
generated.c:5: error: ...
```

### 模板引擎

```c
// template_engine.c 的输出示例
#line 15 "user_template.txt"
    int user_id = get_id();      // 如果出错，指向模板第15行
    char *name = get_name();
#line 20 "user_template.txt"
    process(name);
```

## 实际使用场景

### 1. 断言宏中的行号

```c
// 自定义断言，显示断言所在位置而非宏定义位置
#undef assert
#ifdef NDEBUG
    #define assert(expr) ((void)0)
#else
    #define assert(expr) \
        ((expr) ? (void)0 : \
         assertion_failed(#expr, __FILE__, __LINE__, __func__))
#endif

void assertion_failed(const char *expr, const char *file,
                      int line, const char *func) {
    fprintf(stderr, "断言失败: %s\n文件: %s, 行: %d, 函数: %s\n",
            expr, file, line, func);
    abort();
}
```

### 2. 包含生成的代码

```c
// 在包含自动生成的代码时重置行号
int main(void) {
    // 主逻辑
    process_data();
    
    // 包含自动生成的回调注册代码
#line 1 "generated_callbacks.c"
    #include "generated_callbacks.c"
    
    // 继续主文件的行号
#line 50 "main.c"  // 假设下一行是原文件第50行
    
    start_event_loop();
    return 0;
}
```

### 3. 调试辅助

```c
// 在关键位置插入 #line 帮助定位
void complex_function(void) {
    // 阶段1
#line 1001 "complex_function:phase1"
    initialize();
    
    // 阶段2
#line 2001 "complex_function:phase2"
    process();
    
    // 阶段3
#line 3001 "complex_function:phase3"
    cleanup();
}
// 这样在错误消息中能立即看出出错的阶段
```

## 与 __LINE__ 和 __FILE__ 的交互

```c
#include <stdio.h>

int main(void) {
    printf("行 %d, 文件 %s\n", __LINE__, __FILE__);
    // 行 4, 文件 test.c
    
#line 500 "reset.c"
    printf("行 %d, 文件 %s\n", __LINE__, __FILE__);
    // 行 500, 文件 reset.c
    
    printf("行 %d, 文件 %s\n", __LINE__, __FILE__);
    // 行 501, 文件 reset.c（自动递增）
    
#line 1 "another.c"
    printf("行 %d, 文件 %s\n", __LINE__, __FILE__);
    // 行 1, 文件 another.c
    
    return 0;
}
```

## 使用场景总结

```
场景                      是否常用    说明
─────────────────────────────────────────────────
代码生成器输出              是         指向原始源文件
模板引擎                  是         指向模板文件
调试宏                    偶尔       用于特殊调试目的
包含生成代码              是         重置行号计数
普通编程                  极少       不需要手动修改行号
```

## 限制

```c
// 1. #line 必须产生有效的行号
#line 0        // 错误：行号不能为 0
#line -1       // 错误：行号不能为负

// 2. 文件名必须是字符串字面量
#define MYFILE "test.c"
#line 10 MYFILE  // 某些编译器可能不支持这种写法

// 3. #line 影响错误报告和调试信息，但不影响实际的文件内容
```

## 重要注意事项

> **关键点总结**：
> 1. `#line` 改变 `__LINE__` 和 `__FILE__` 的值，影响编译器的错误消息
> 2. 主要用于**代码生成器**——让错误指向原始源文件
> 3. 行号从指定值开始自动递增
> 4. 文件名修改是局部的，不会影响其他源文件
> 5. 在日常编程中很少直接使用 `#line`
> 6. 行号必须为正整数，文件名必须是字符串字面量
