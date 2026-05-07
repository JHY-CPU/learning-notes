# CMake构建系统入门

## 一、概念说明

CMake是一个**跨平台**的构建系统生成器。它不直接编译代码，而是根据`CMakeLists.txt`配置文件生成对应平台的构建文件（Makefile、Visual Studio项目、Xcode项目等）。

## 二、具体用法

### 2.1 最简单的CMakeLists.txt

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)  # 最低CMake版本
project(HelloWorld)                     # 项目名称

set(CMAKE_CXX_STANDARD 17)            # C++标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)    # 强制要求标准版本

add_executable(hello main.cpp)         # 生成可执行文件
```

```cpp
// main.cpp
#include <iostream>
int main() {
    std::cout << "Hello CMake!" << std::endl;
    return 0;
}
```

```bash
# 构建步骤
mkdir build && cd build
cmake ..
make
./hello
```

输出：
```
Hello CMake!
```

### 2.2 多文件项目

```cmake
# 项目结构：
# ├── CMakeLists.txt
# ├── include/
# │   └── utils.h
# ├── src/
# │   ├── main.cpp
# │   └── utils.cpp
# └── build/

cmake_minimum_required(VERSION 3.20)
project(MultiFileProject)

set(CMAKE_CXX_STANDARD 17)

# 指定头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 收集源文件
set(SOURCES
    src/main.cpp
    src/utils.cpp
)

add_executable(app ${SOURCES})
```

### 2.3 使用库

```cmake
cmake_minimum_required(VERSION 3.20)
project(LibraryProject)

set(CMAKE_CXX_STANDARD 17)

# 创建静态库
add_library(mathlib STATIC
    src/math/add.cpp
    src/math/multiply.cpp
)

# 指定库的头文件路径
target_include_directories(mathlib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 创建可执行文件并链接库
add_executable(calculator src/main.cpp)
target_link_libraries(calculator PRIVATE mathlib)
```

### 2.4 查找和使用第三方库

```cmake
cmake_minimum_required(VERSION 3.20)
project(ThirdPartyDemo)

set(CMAKE_CXX_STANDARD 17)

# 查找已安装的库
find_package(Threads REQUIRED)      # pthread
find_package(OpenSSL REQUIRED)      # OpenSSL

add_executable(app main.cpp)

# 链接第三方库
target_link_libraries(app
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
)
```

### 2.5 常用CMake变量和命令

```cmake
# 预定义变量
${PROJECT_SOURCE_DIR}    # 项目根目录
${PROJECT_BINARY_DIR}    # 构建目录
${CMAKE_CURRENT_SOURCE_DIR}  # 当前CMakeLists.txt所在目录

# 编译选项
set(CMAKE_BUILD_TYPE "Release")  # Debug/Release/MinSizeRel/RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 条件编译
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_MODE)
endif()

# 安装规则
install(TARGETS app DESTINATION bin)
install(FILES include/mylib.h DESTINATION include)
```

## 三、注意事项与常见陷阱

1. **外部构建**：永远不要在源码目录中直接运行cmake，使用`mkdir build && cd build && cmake ..`
2. **CMake版本**：不同版本的CMake行为可能不同，始终指定`cmake_minimum_required`
3. **PUBLIC/PRIVATE/INTERFACE**：`target_include_directories`和`target_link_libraries`中这三个关键字控制依赖传递
4. **不要用include_directories**：优先使用`target_include_directories`，它是目标级别的更精确控制
5. **CMake缓存**：修改CMakeLists.txt后需要重新运行cmake，变量值可能被缓存
