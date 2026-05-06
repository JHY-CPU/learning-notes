# CLion开发环境配置

## 一、CLion简介

CLion是JetBrains公司开发的跨平台C/C++集成开发环境，基于IntelliJ平台，提供智能代码补全、强大的重构功能和集成调试。

### 1.1 主要特点

- 智能代码补全和代码分析
- 集成CMake构建系统
- 强大的调试器（GDB/LLDB/MSVC）
- 代码重构（重命名、提取函数等）
- 内置终端和版本控制
- 支持多种编译器（GCC、Clang、MSVC）

### 1.2 获取方式

- **官方网址**：https://www.jetbrains.com/clion/
- **学生授权**：免费（需验证学生身份）
- **试用**：30天免费试用
- **社区版**：不提供

## 二、初始配置

### 2.1 首次启动配置

1. 启动CLion
2. 选择UI主题（Darcula/Light）
3. 配置工具链：
   - CLion会自动检测已安装的编译器和调试器
   - Windows上需要MinGW或WSL
   - Linux/macOS通常自动检测到GCC/Clang

### 2.2 工具链配置

**File → Settings → Build, Execution, Deployment → Toolchains**

| 平台 | 推荐配置 |
|------|---------|
| Windows | MinGW + GDB 或 WSL |
| Linux | GCC + GDB |
| macOS | Clang + LLDB |

**Windows MinGW配置**：
```
Toolchain:
  Name: MinGW
  CMake: Bundled (自带)
  C Compiler: C:\msys64\mingw64\bin\gcc.exe
  C++ Compiler: C:\msys64\mingw64\bin\g++.exe
  Debugger: C:\msys64\mingw64\bin\gdb.exe
```

**WSL配置**（Windows）：
```
Toolchain:
  Name: WSL
  CMake: /usr/bin/cmake
  C Compiler: /usr/bin/gcc
  C++ Compiler: /usr/bin/g++
  Debugger: /usr/bin/gdb
```

## 三、创建项目

### 3.1 创建C项目

1. **File → New Project**
2. 选择 **C Executable**
3. 选择C标准（C11/C17）
4. 设置项目名称和位置
5. 点击 **Create**

CLion会自动生成以下文件：

```
project/
├── CMakeLists.txt
├── main.c
└── cmake-build-debug/
```

### 3.2 生成的CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project C)

set(CMAKE_C_STANDARD 11)

add_executable(my_project main.c)
```

### 3.3 生成的main.c

```c
#include <stdio.h>

int main(void) {
    printf("Hello, CLion!\n");
    return 0;
}
```

## 四、CMake配置

### 4.1 基本CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project C)

# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 编译选项
add_compile_options(-Wall -Wextra)

# Debug/Release配置
set(CMAKE_C_FLAGS_DEBUG "-g -Og -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")

# 添加源文件
set(SOURCES
    src/main.c
    src/utils.c
    src/math_ops.c
)

# 创建可执行文件
add_executable(my_project ${SOURCES})

# 包含目录
target_include_directories(my_project PRIVATE include)

# 链接库
target_link_libraries(my_project m)  # 链接数学库
```

### 4.2 多目录项目结构

```
project/
├── CMakeLists.txt            （顶层CMake）
├── src/
│   ├── CMakeLists.txt        （src目录CMake）
│   ├── main.c
│   └── app.c
├── lib/
│   ├── CMakeLists.txt        （库目录CMake）
│   ├── mylib.c
│   └── mylib.h
├── include/
│   └── mylib.h
└── tests/
    ├── CMakeLists.txt
    └── test_mylib.c
```

**顶层CMakeLists.txt**：
```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project C)

set(CMAKE_C_STANDARD 11)

# 添加子目录
add_subdirectory(lib)
add_subdirectory(src)

# 测试（可选）
enable_testing()
add_subdirectory(tests)
```

**lib/CMakeLists.txt**：
```cmake
# 创建静态库
add_library(mylib STATIC mylib.c)
target_include_directories(mylib PUBLIC ${CMAKE_SOURCE_DIR}/include)
```

**src/CMakeLists.txt**：
```cmake
add_executable(my_project main.c app.c)
target_include_directories(my_project PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(my_project mylib)
```

### 4.3 使用find_package

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project C)

set(CMAKE_C_STANDARD 11)

# 查找系统库
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

add_executable(my_project main.c)
target_include_directories(my_project PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(my_project ${SDL2_LIBRARIES})
```

## 五、编写代码

### 5.1 代码编辑功能

| 功能 | 快捷键（Windows/Linux） | 快捷键（macOS） |
|------|----------------------|----------------|
| 代码补全 | `Ctrl + Space` | `Ctrl + Space` |
| 智能补全 | `Ctrl + Shift + Space` | `Ctrl + Shift + Space` |
| 快速文档 | `Ctrl + Q` | `F1` |
| 跳转到定义 | `Ctrl + B` | `Cmd + B` |
| 查找用法 | `Alt + F7` | `Alt + F7` |
| 重命名 | `Shift + F6` | `Shift + F6` |
| 提取变量 | `Ctrl + Alt + V` | `Cmd + Alt + V` |
| 提取函数 | `Ctrl + Alt + M` | `Cmd + Alt + M` |
| 格式化代码 | `Ctrl + Alt + L` | `Cmd + Alt + L` |
| 注释 | `Ctrl + /` | `Cmd + /` |

### 5.2 Live Templates

CLion提供代码模板，快速生成常用代码片段：

输入 `main` + Tab：
```c
int main(int argc, char *argv[]) {
    return 0;
}
```

输入 `for` + Tab：
```c
for (int i = 0; i < ; ++i) {

}
```

输入 `guard` + Tab（在头文件中）：
```c
#ifndef PROJECT_FILENAME_H
#define PROJECT_FILENAME_H

#endif //PROJECT_FILENAME_H
```

### 5.3 自定义Live Templates

**File → Settings → Editor → Live Templates**

例如，添加调试打印模板：
```
Abbreviation: dbg
Template text: printf("[DEBUG] %s:%d - $VAR$ = %$FORMAT$\n", __FILE__, __LINE__, $VAR$);
```

## 六、编译运行

### 6.1 构建配置

**Run → Edit Configurations**

| 配置项 | 说明 |
|--------|------|
| Target | 选择要构建的目标 |
| Executable | 可执行文件路径 |
| Program arguments | 命令行参数 |
| Working directory | 工作目录 |
| Environment variables | 环境变量 |

### 6.2 构建和运行

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 构建项目 | `Ctrl + F9` | 编译所有文件 |
| 运行 | `Shift + F10` | 运行程序 |
| 调试运行 | `Shift + F9` | 以调试模式运行 |
| 重新构建 | `Ctrl + Shift + F9` | 清理并重新构建 |
| 运行配置 | `Alt + Shift + F10` | 选择运行配置 |

### 6.3 CMake操作

在 **View → Tool Windows → CMake** 中可以：

- 重新加载CMake项目
- 查看CMake输出
- 清理缓存并重新加载
- 选择CMake配置（Debug/Release/RelWithDebInfo）

## 七、调试

### 7.1 断点管理

- **行断点**：点击行号左侧
- **条件断点**：右键断点 → 输入条件
- **日志断点**：右键断点 → Log evaluated expression
- **临时断点**：右键行号 → Toggle Temporary Line Breakpoint

### 7.2 调试操作

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 开始调试 | `Shift + F9` | 启动调试 |
| 继续 | `F9` | 继续执行 |
| 逐过程 | `F8` | Step Over |
| 逐语句 | `F7` | Step Into |
| 跳出 | `Shift + F8` | Step Out |
| 运行到光标 | `Alt + F9` | 执行到光标位置 |
| 评估表达式 | `Alt + F8` | 计算表达式值 |

### 7.3 调试窗口

**Debug窗口中的面板**：

- **Frames**：调用堆栈
- **Variables**：变量值
- **Watches**：监视表达式
- **Memory**：内存查看
- **Console**：程序输出
- **Debugger**：直接输入GDB/LLDB命令

### 7.4 内存视图

在调试时查看内存：
1. 右键变量 → **View Memory**
2. 可以查看十六进制内存内容
3. 支持修改内存值

## 八、代码分析

### 8.1 静态分析

CLion内置了静态代码分析，实时检测问题：

- 未使用的变量
- 内存泄漏（简单场景）
- 数组越界（部分场景）
- 空指针解引用
- 类型不匹配
- 死代码

### 8.2 Clang-Tidy集成

**File → Settings → Editor → Inspections → Clang-Tidy**

可以在项目根目录创建 `.clang-tidy` 文件：

```yaml
Checks: >
  -*,
  clang-analyzer-*,
  bugprone-*,
  readability-*,
  modernize-*,
  performance-*,
  portability-*,
  -bugprone-easily-swappable-parameters

CheckOptions:
  - key: readability-identifier-naming.VariableCase
    value: lower_case
  - key: readability-identifier-naming.FunctionCase
    value: lower_case
```

### 8.3 代码清理

**Code → Code Cleanup** 自动修复代码风格问题。

## 九、版本控制集成

### 9.1 Git集成

CLion内置Git支持：

| 操作 | 菜单/快捷键 |
|------|------------|
| 提交 | `Ctrl + K` |
| 推送 | `Ctrl + Shift + K` |
| 拉取 | `VCS → Git → Pull` |
| 分支 | 右下角分支名 |
| 日志 | `Alt + 9` → Log |
| 对比 | `Ctrl + D` |

### 9.2 .gitignore

```gitignore
# CLion项目
cmake-build-*/
.idea/
*.iml

# 编译输出
*.o
*.a
*.so
*.exe
```

## 十、实用技巧

### 10.1 自定义编译选项

在CMakeLists.txt中添加条件编译：

```cmake
# 根据构建类型设置不同选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -DDEBUG -Wall -Wextra)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2 -DNDEBUG -Wall -Wextra)
endif()
```

### 10.2 使用外部工具

**File → Settings → Tools → External Tools**

添加常用命令，如：
- `make clean`
- `valgrind` 内存检查
- `cppcheck` 静态分析

### 10.3 键盘快捷键推荐

| 操作 | Windows/Linux | macOS |
|------|-------------|-------|
| 全部保存 | `Ctrl + S` | `Cmd + S` |
| 查找文件 | `Ctrl + Shift + N` | `Cmd + Shift + O` |
| 查找符号 | `Ctrl + Alt + Shift + N` | `Cmd + Alt + O` |
| 最近文件 | `Ctrl + E` | `Cmd + E` |
| 切换头/源 | `Ctrl + Alt + Home` | `Ctrl + Shift + Up` |
| 最大化窗口 | `Ctrl + Shift + F12` | `Cmd + Shift + F12` |

## 十一、常见问题

### 11.1 CMake找不到编译器

检查 **Settings → Build → Toolchains** 中的编译器路径是否正确。

### 11.2 构建速度慢

- 使用Ninja生成器：`cmake -G Ninja`
- 在CMake配置中使用 `set(CMAKE_BUILD_PARALLEL_LEVEL 8)`
- CLion设置中启用增量编译

### 11.3 调试器无法启动

- 确保编译时使用了 `-g` 选项
- 检查调试器路径配置
- Windows上确认GDB版本与GCC匹配

## 十二、关键要点

> **重要提示**：
> 1. CLion基于CMake构建系统，项目以CMakeLists.txt为核心
> 2. 工具链配置是关键：确保编译器和调试器路径正确
> 3. 智能补全和重构功能是CLion的优势
> 4. 快捷键：`Shift + F10`运行，`Shift + F9`调试
> 5. 学生可以申请免费授权
> 6. Live Templates可以大幅提高编码效率
> 7. 内置Clang-Tidy静态分析有助于代码质量
