# CMake入门

## 一、CMake概述

CMake是一个跨平台的构建系统生成器，它不直接编译代码，而是生成适合各平台的构建文件（如Makefile、Visual Studio项目、Ninja文件等）。

### 1.1 CMake工作流程

```
CMakeLists.txt → CMake → Makefile/VS项目/Ninja → 构建工具 → 可执行文件
```

```bash
# 典型的CMake使用流程
mkdir build && cd build
cmake ..
make          # Linux/macOS
# 或 cmake --build .   跨平台方式
```

### 1.2 安装CMake

```bash
# Ubuntu/Debian
sudo apt install cmake

# CentOS/RHEL
sudo yum install cmake

# macOS
brew install cmake

# Windows
# 下载安装包: https://cmake.org/download/
```

## 二、第一个CMake项目

### 2.1 最简单的CMakeLists.txt

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(hello C)

add_executable(hello main.c)
```

```c
/* main.c */
#include <stdio.h>

int main(void) {
    printf("Hello, CMake!\n");
    return 0;
}
```

```bash
# 构建步骤
mkdir build
cd build
cmake ..
cmake --build .
./hello
```

### 2.2 CMake基本命令

```cmake
# cmake_minimum_required: 指定最低CMake版本
cmake_minimum_required(VERSION 3.20)

# project: 定义项目名称和语言
project(myproject C CXX)  # 支持C和C++

# add_executable: 创建可执行文件
add_executable(myapp main.c utils.c)

# set: 设置变量
set(SOURCES main.c utils.c math.c)
add_executable(myapp ${SOURCES})
```

## 三、项目结构

### 3.1 典型项目布局

```
myproject/
├── CMakeLists.txt        # 顶层CMake
├── src/
│   ├── CMakeLists.txt    # src目录的CMake
│   ├── main.c
│   └── app.c
├── include/
│   └── myproject/
│       └── app.h
├── lib/
│   ├── CMakeLists.txt
│   ├── mylib.c
│   └── mylib.h
├── tests/
│   ├── CMakeLists.txt
│   └── test_app.c
└── build/
```

### 3.2 多文件项目

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(myproject C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 收集源文件
set(SOURCES
    src/main.c
    src/app.c
    src/utils.c
)

# 头文件目录
set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
)

# 创建可执行文件
add_executable(myproject ${SOURCES})

# 包含头文件目录
target_include_directories(myproject PRIVATE ${INCLUDE_DIRS})

# 链接库
target_link_libraries(myproject m)  # 链接数学库
```

## 四、常用CMake命令

### 4.1 项目设置

```cmake
# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)    # 强制要求该标准
set(CMAKE_C_EXTENSIONS OFF)          # 禁用GNU扩展

# 设置编译选项
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")

# 推荐方式：使用target命令
target_compile_options(myapp PRIVATE -Wall -Wextra)
```

### 4.2 添加源文件

```cmake
# 方式1：直接列出
add_executable(myapp main.c utils.c math.c)

# 方式2：使用变量
set(SRCS main.c utils.c math.c)
add_executable(myapp ${SRCS})

# 方式3：使用通配符（不推荐用于大项目）
file(GLOB SRCS "src/*.c")
add_executable(myapp ${SRCS})

# 方式4：递归通配（更不推荐）
file(GLOB_RECURSE SRCS "src/*.c")
```

### 4.3 包含目录

```cmake
# target_include_directories的可见性
# PRIVATE: 仅当前目标使用
# INTERFACE: 仅依赖此目标的其他目标使用
# PUBLIC: 当前目标和依赖目标都使用

target_include_directories(myapp
    PRIVATE ${CMAKE_SOURCE_DIR}/src
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

# 旧方式（全局设置，不推荐）
include_directories(${CMAKE_SOURCE_DIR}/include)
```

### 4.4 链接库

```cmake
# 链接系统库
target_link_libraries(myapp m pthread)

# 链接自定义库
target_link_libraries(myapp mylib)

# 链接静态库
target_link_libraries(myapp ${CMAKE_SOURCE_DIR}/lib/libhelper.a)

# 链接共享库
find_library(MYLIB mylib PATHS /usr/local/lib)
target_link_libraries(myapp ${MYLIB})
```

## 五、构建库

### 5.1 静态库

```cmake
# 创建静态库
add_library(mylib STATIC
    lib/mylib.c
    lib/helper.c
)

# 设置库的包含目录
target_include_directories(mylib
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

# 使用库
add_executable(myapp main.c)
target_link_libraries(myapp mylib)
```

### 5.2 动态库

```cmake
# 创建动态库
add_library(mylib SHARED
    lib/mylib.c
    lib/helper.c
)

# 设置库版本
set_target_properties(mylib PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# 设置位置无关代码
set_target_properties(mylib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# 使用库
add_executable(myapp main.c)
target_link_libraries(myapp mylib)
```

## 六、条件编译

### 6.1 选项

```cmake
# 定义选项（可通过-D传递）
option(ENABLE_DEBUG "Enable debug mode" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# 使用选项
if(ENABLE_DEBUG)
    add_definitions(-DDEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

```bash
# 通过命令行设置选项
cmake -DENABLE_DEBUG=ON -DBUILD_TESTS=OFF ..
```

### 6.2 构建类型

```cmake
# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 根据构建类型设置选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2 -DNDEBUG)
endif()
```

```bash
# 指定构建类型
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## 七、查找外部库

### 7.1 find_package

```cmake
# 查找系统库
find_package(PkgConfig REQUIRED)

# 使用pkg-config查找库
pkg_check_modules(SDL2 REQUIRED sdl2)

target_include_directories(myapp PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(myapp ${SDL2_LIBRARIES})
```

### 7.2 find_library

```cmake
# 查找库文件
find_library(MATH_LIB m)
find_library(PTHREAD_LIB pthread)

if(MATH_LIB)
    target_link_libraries(myapp ${MATH_LIB})
endif()
```

## 八、子目录

### 8.1 add_subdirectory

```cmake
# 顶层 CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(myproject C)

set(CMAKE_C_STANDARD 11)

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(tests)
```

```cmake
# src/CMakeLists.txt
add_executable(myapp main.c)
target_link_libraries(myapp mylib)
```

```cmake
# lib/CMakeLists.txt
add_library(mylib STATIC mylib.c)
target_include_directories(mylib PUBLIC ${CMAKE_SOURCE_DIR}/include)
```

## 九、安装

```cmake
# 安装目标
install(TARGETS myapp
    RUNTIME DESTINATION bin
)

install(TARGETS mylib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 安装头文件
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 安装配置文件
install(FILES config.ini
    DESTINATION etc/myapp
)
```

```bash
# 执行安装
cmake --install . --prefix /usr/local
# 或
make install DESTDIR=/tmp/package
```

## 十、测试（CTest）

```cmake
# 启用测试
enable_testing()

# 添加测试
add_executable(test_math tests/test_math.c)
target_link_libraries(test_math mylib)

add_test(NAME test_math COMMAND test_math)

# 多个测试
foreach(test_name test_add test_subtract test_multiply)
    add_executable(${test_name} tests/${test_name}.c)
    target_link_libraries(${test_name} mylib)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
```

```bash
# 运行测试
ctest
ctest --verbose
ctest -R "math"      # 运行名字包含math的测试
```

## 十一、完整示例

```cmake
cmake_minimum_required(VERSION 3.20)
project(calculator C)

# 编译器设置
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# 选项
option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
add_compile_options(-Wall -Wextra)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
endif()

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# 源文件
set(LIB_SOURCES src/calc.c src/utils.c)
set(APP_SOURCES src/main.c)

# 创建库
add_library(calclib STATIC ${LIB_SOURCES})
target_include_directories(calclib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 创建可执行文件
add_executable(calculator ${APP_SOURCES})
target_link_libraries(calculator calclib m)

# 测试
if(BUILD_TESTS)
    enable_testing()
    file(GLOB TEST_SOURCES tests/test_*.c)
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} calclib)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# 安装
install(TARGETS calculator DESTINATION bin)
install(TARGETS calclib DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
```

## 十二、关键要点

> **重要提示**：
> 1. CMake生成构建文件，不直接编译
> 2. `cmake_minimum_required` 必须放在第一行
> 3. 使用 `target_xxx` 命令而非全局命令（更好的封装性）
> 4. `PRIVATE`/`PUBLIC`/`INTERFACE` 控制依赖可见性
> 5. 使用 `-D` 传递变量给CMake：`cmake -DVAR=value ..`
> 6. 推荐out-of-source构建：`mkdir build && cd build && cmake ..`
> 7. `enable_testing()` + `add_test()` 配合CTest使用
> 8. 使用 `option()` 定义可配置的布尔选项
> 9. `set_target_properties` 设置目标的特殊属性
> 10. CMakeLists.txt中的命令不区分大小写（推荐小写）
