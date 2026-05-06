# Visual Studio开发环境

## 一、Visual Studio简介

Visual Studio是微软开发的集成开发环境（IDE），是Windows平台上最强大的开发工具之一。

### 1.1 版本选择

| 版本 | 适用场景 | 费用 |
|------|---------|------|
| Community | 个人和小型团队 | 免费 |
| Professional | 专业开发 | 付费 |
| Enterprise | 企业级开发 | 付费 |

**推荐**：学习C语言使用 **Visual Studio Community** 版本即可。

### 1.2 下载安装

1. 访问 https://visualstudio.microsoft.com/
2. 下载 Visual Studio Community
3. 运行安装程序
4. 选择工作负载：**使用C++的桌面开发**
5. 点击安装

> **注意**：虽然选择的是C++工作负载，但它完全支持C语言开发。

## 二、创建C项目

### 2.1 创建新项目

1. 打开Visual Studio
2. 选择 **创建新项目**
3. 搜索并选择 **空项目**
4. 点击 **下一步**
5. 输入项目名称，选择位置
6. 点击 **创建**

### 2.2 添加C源文件

1. 在 **解决方案资源管理器** 中，右键点击 **源文件**
2. 选择 **添加** → **新建项**
3. 选择 **C++文件(.cpp)**
4. 将文件名改为以 `.c` 结尾（如 `main.c`）
5. 点击 **添加**

> **关键**：文件扩展名必须是 `.c`，Visual Studio会根据扩展名决定使用C编译器还是C++编译器。

### 2.3 编写代码

```c
// main.c
#include <stdio.h>

int main(void) {
    printf("Hello from Visual Studio!\n");
    return 0;
}
```

### 2.4 编译运行

- **编译**：`Ctrl + Shift + B`
- **运行（不调试）**：`Ctrl + F5`
- **调试运行**：`F5`

## 三、项目配置

### 3.1 打开项目属性

右键点击项目 → **属性**（或 `Alt + Enter`）

### 3.2 常用配置选项

**配置选择**：
- **Debug**：调试版本，包含调试信息，无优化
- **Release**：发布版本，有优化，无调试信息

**C/C++ 常规设置**：

```
C/C++ → 常规 → 附加包含目录
  添加自定义头文件路径

C/C++ → 常规 → 警告等级
  推荐设置为：Level4 (/W4)

C/C++ → 常规 → SDL检查
  推荐：是 (/sdl)
```

**C/C++ 预处理器**：

```
C/C++ → 预处理器 → 预处理器定义
  _CRT_SECURE_NO_WARNINGS    （禁用安全函数警告）
  DEBUG                      （调试宏）
```

**C/C++ 语言**：

```
C/C++ → 语言 → C语言标准
  C17 (/std:c17)    推荐
  C11 (/std:c11)
  C99 (/std:c99)
```

**C/C++ 优化**：

```
C/C++ → 优化 → 优化
  Debug:   已禁用 (/Od)
  Release: 最大优化速度 (/O2)

C/C++ → 优化 → 内联函数扩展
  Debug:   已禁用 (/Ob0)
  Release: 任何适用的 (/Ob2)
```

**C/C++ 代码生成**：

```
C/C++ → 代码生成 → 运行库
  Debug:   多线程调试 (/MTd)
  Release: 多线程 (/MT)

C/C++ → 代码生成 → 安全检查
  Debug:   启用安全检查 (/GS)
  Release: 启用安全检查 (/GS)
```

### 3.3 链接器配置

```
链接器 → 常规 → 附加库目录
  添加库文件搜索路径

链接器 → 输入 → 附加依赖项
  添加要链接的库文件
  例如：ws2_32.lib;winmm.lib;

链接器 → 系统 → 子系统
  控制台 (/SUBSYSTEM:CONSOLE)
```

## 四、调试功能

### 4.1 断点

| 操作 | 方法 |
|------|------|
| 设置断点 | 点击行号左侧灰色区域，或 `F9` |
| 条件断点 | 右键断点 → 条件 |
| 删除断点 | 再次点击，或 `Ctrl + Shift + F9` |
| 禁用断点 | 右键断点 → 禁用断点 |

### 4.2 调试控制

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 开始调试 | `F5` | 启动或继续调试 |
| 停止调试 | `Shift + F5` | 终止程序 |
| 逐过程 | `F10` | Step Over |
| 逐语句 | `F11` | Step Into |
| 跳出 | `Shift + F11` | Step Out |
| 运行到光标处 | `Ctrl + F10` | Run to Cursor |

### 4.3 调试窗口

```
调试 → 窗口 → 断点        管理所有断点
调试 → 窗口 → 即时窗口     执行表达式
调试 → 窗口 → 监视        添加变量监视
调试 → 窗口 → 局部变量    查看局部变量
调试 → 窗口 → 自动变量    自动显示当前行变量
调试 → 窗口 → 调用堆栈    查看函数调用链
调试 → 窗口 → 内存        查看内存内容
调试 → 窗口 → 寄存器      查看CPU寄存器
调试 → 窗口 → 反汇编      查看汇编代码
```

### 4.4 数据断点（仅x86/x64）

数据断点在变量值改变时触发：

1. 在调试模式下，打开 **调试 → 新建断点 → 数据断点**
2. 输入要监视的内存地址
3. 设置监视的字节数（1、2、4、8字节）

## 五、常用功能

### 5.1 代码导航

| 功能 | 快捷键 | 说明 |
|------|--------|------|
| 转到定义 | `F12` | 跳转到函数/变量定义 |
| 转到声明 | `Ctrl + F12` | 跳转到声明 |
| 查找所有引用 | `Shift + F12` | 查找所有使用位置 |
| 转到行 | `Ctrl + G` | 跳转到指定行 |
| 转到文件 | `Ctrl + ,` | 快速打开文件 |
| 后退导航 | `Ctrl + -` | 返回上一个位置 |
| 前进导航 | `Ctrl + Shift + -` | 前进到下一个位置 |

### 5.2 代码编辑

| 功能 | 快捷键 | 说明 |
|------|--------|------|
| 格式化代码 | `Ctrl + K, Ctrl + D` | 格式化整个文档 |
| 格式化选中 | `Ctrl + K, Ctrl + F` | 格式化选中代码 |
| 注释 | `Ctrl + K, Ctrl + C` | 注释选中行 |
| 取消注释 | `Ctrl + K, Ctrl + U` | 取消注释 |
| 智能感知 | `Ctrl + J` | 触发自动完成 |
| 快速信息 | `Ctrl + K, Ctrl + I` | 显示工具提示 |
| 重命名 | `Ctrl + R, Ctrl + R` | 重命名符号 |

### 5.3 查找和替换

| 功能 | 快捷键 |
|------|--------|
| 查找 | `Ctrl + F` |
| 替换 | `Ctrl + H` |
| 在文件中查找 | `Ctrl + Shift + F` |
| 在文件中替换 | `Ctrl + Shift + H` |

## 六、多文件项目示例

### 6.1 项目结构

```
MyProject/
├── MyProject.sln          （解决方案文件）
├── MyProject/             （项目文件夹）
│   ├── MyProject.vcxproj  （项目文件）
│   ├── main.c             （主文件）
│   ├── utils.h            （工具函数头文件）
│   ├── utils.c            （工具函数实现）
│   ├── math_ops.h         （数学运算头文件）
│   └── math_ops.c         （数学运算实现）
```

### 6.2 代码示例

```c
/* math_ops.h */
#ifndef MATH_OPS_H
#define MATH_OPS_H

int add(int a, int b);
int multiply(int a, int b);

#endif
```

```c
/* math_ops.c */
#include "math_ops.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
```

```c
/* main.c */
#include <stdio.h>
#include "math_ops.h"

int main(void) {
    printf("3 + 4 = %d\n", add(3, 4));
    printf("3 * 4 = %d\n", multiply(3, 4));
    return 0;
}
```

## 七、常见问题解决

### 7.1 `scanf` 安全警告

```
warning C4996: 'scanf': This function or variable may be unsafe.
```

**解决方法**：

方法1：在文件开头添加宏定义
```c
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
```

方法2：在项目属性中添加预处理器定义
```
C/C++ → 预处理器 → 预处理器定义
添加：_CRT_SECURE_NO_WARNINGS
```

方法3：使用安全版本函数
```c
scanf_s("%d", &x);  /* MSVC专用的"安全"版本 */
```

### 7.2 控制台窗口闪退

在 `return 0;` 前添加：
```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("Hello\n");

    /* 防止窗口关闭 */
    printf("按回车键退出...");
    getchar();

    return 0;
}
```

或使用 `Ctrl + F5`（开始执行不调试）运行。

### 7.3 中文乱码

```
文件 → 高级保存选项 → 编码
选择：Unicode (UTF-8 带签名) - 代码页 65001
```

或在项目属性中：
```
C/C++ → 所有选项 → 附加选项
添加：/utf-8
```

### 7.4 找不到头文件

检查项目属性中的包含目录：
```
C/C++ → 常规 → 附加包含目录
添加头文件所在目录
```

## 八、项目模板配置

### 8.1 导出模板

配置好项目后，可以导出为模板：

1. **项目** → **导出模板**
2. 选择 **项目模板**
3. 填写模板名称和描述
4. 点击 **完成**

以后创建新项目时，可以选择该模板。

### 8.2 推荐的项目模板设置

```
项目属性推荐配置：

C/C++ → 常规:
  - 警告等级: Level4 (/W4)
  - SDL检查: 是 (/sdl)
  - C语言标准: C17 (/std:c17)

C/C++ → 预处理器:
  - 预处理器定义: _CRT_SECURE_NO_WARNINGS

C/C++ → 代码生成:
  - 运行库: 多线程调试 (/MTd)
  - 安全检查: 启用安全检查 (/GS)

链接器 → 系统:
  - 子系统: 控制台 (/SUBSYSTEM:CONSOLE)
```

## 九、关键要点

> **重要提示**：
> 1. Visual Studio Community是免费的，适合学习使用
> 2. 源文件扩展名必须是 `.c`（不是 `.cpp`）
> 3. 使用 `F5` 调试运行，`Ctrl + F5` 不调试运行
> 4. 定义 `_CRT_SECURE_NO_WARNINGS` 消除安全函数警告
> 5. Debug配置用于开发调试，Release配置用于发布
> 6. F12转到定义，F9设置断点，F10逐过程，F11逐语句
> 7. 推荐设置警告等级为 `/W4`
