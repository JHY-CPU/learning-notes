# 包管理器 vcpkg

## 一、概念说明

vcpkg是微软开发的**C/C++包管理器**，支持Windows、Linux和macOS。它简化了第三方库的安装和管理，解决了C++生态中长期缺乏统一包管理器的问题。

## 二、具体用法

### 2.1 安装vcpkg

```bash
# 克隆vcpkg仓库
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

# Windows
.\bootstrap-vcpkg.bat

# Linux/macOS
./bootstrap-vcpkg.sh

# 将vcpkg添加到PATH（可选）
export PATH="$PATH:$(pwd)"
```

### 2.2 搜索和安装库

```bash
# 搜索库
vcpkg search boost
vcpkg search json

# 安装库
vcpkg install nlohmann-json
vcpkg install fmt
vcpkg install boost-system

# 安装特定平台的库
vcpkg install openssl:x64-windows
vcpkg install zlib:x64-linux

# 查看已安装的库
vcpkg list

# 升级所有库
vcpkg upgrade --no-dry-run

# 卸载库
vcpkg remove nlohmann-json
```

### 2.3 与CMake集成

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(VcpkgDemo)

set(CMAKE_CXX_STANDARD 17)

# vcpkg会自动设置CMAKE_PREFIX_PATH
find_package(nlohmann_json CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(app main.cpp)

target_link_libraries(app
    PRIVATE
    nlohmann_json::nlohmann_json
    fmt::fmt
)
```

```cpp
// main.cpp
#include <nlohmann/json.hpp>
#include <fmt/core.h>

int main() {
    // 使用nlohmann-json库
    nlohmann::json j = {
        {"name", "张三"},
        {"age", 25},
        {"skills", {"C++", "Python", "Rust"}}
    };
    fmt::print("JSON: {}\n", j.dump(2));
    return 0;
}
```

输出：
```
JSON: {
  "age": 25,
  "name": "张三",
  "skills": [
    "C++",
    "Python",
    "Rust"
  ]
}
```

### 2.4 manifest模式（推荐）

```json
// vcpkg.json —— 项目根目录
{
    "name": "my-project",
    "version-string": "1.0.0",
    "dependencies": [
        "nlohmann-json",
        "fmt",
        {
            "name": "openssl",
            "platform": "!windows"
        }
    ]
}
```

```bash
# CMake时自动安装依赖
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

### 2.5 自定义端口

```cmake
# ports/mylib/portfile.cmake
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO author/mylib
    REF v1.0.0
    SHA512 abc123...
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
```

## 三、注意事项与常见陷阱

1. **三元组**：vcpkg使用三元组（如`x64-windows`）指定目标平台，注意选择正确的三元组
2. **CMAKE_TOOLCHAIN_FILE**：必须在第一次cmake配置时指定工具链文件，之后不能更改
3. **manifest模式**：新项目推荐使用manifest模式（vcpkg.json），管理依赖更清晰
4. **磁盘空间**：vcpkg会下载源码并本地编译，可能占用大量磁盘空间
5. **版本固定**：vcpkg默认不锁定版本，需要使用baseline机制固定版本
