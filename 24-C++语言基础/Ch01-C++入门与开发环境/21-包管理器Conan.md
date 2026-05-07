# 包管理器 Conan

## 一、概念说明

Conan是一个**去中心化**的C/C++包管理器，支持所有平台和构建系统。它使用Python编写的配方（recipe）来定义包，具有灵活的依赖解析和版本管理能力。

## 二、具体用法

### 2.1 安装Conan

```bash
# 使用pip安装（推荐）
pip install conan

# 检查版本
conan --version

# 首次使用配置
conan profile detect --force
```

### 2.2 conanfile.txt方式

```ini
# conanfile.txt —— 最简单的依赖声明
[requires]
zlib/1.3.1
openssl/3.2.0
nlohmann_json/3.11.3

[generators]
CMakeDeps
CMakeToolchain

[options]
openssl/*:shared=True
```

```bash
# 安装依赖
conan install . --output-folder=build --build=missing

# 使用CMake构建
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
```

### 2.3 conanfile.py方式（推荐）

```python
# conanfile.py —— 更灵活的方式
from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout

class MyProject(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("zlib/1.3.1")
        self.requires("openssl/3.2.0")
        self.requires("fmt/10.2.1")

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
```

### 2.4 与CMake集成

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(ConanDemo)

set(CMAKE_CXX_STANDARD 17)

find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(fmt REQUIRED)

add_executable(app main.cpp)

target_link_libraries(app
    PRIVATE
    ZLIB::ZLIB
    OpenSSL::SSL
    fmt::fmt
)
```

```cpp
// main.cpp
#include <fmt/core.h>
#include <zlib.h>

int main() {
    fmt::print("zlib版本: {}\n", zlibVersion());
    return 0;
}
```

输出：
```
zlib版本: 1.3.1
```

### 2.5 创建和发布自己的包

```python
# conanfile.py（库的配方）
from conan import ConanFile
from conan.tools.cmake import CMake

class MyLibConan(ConanFile):
    name = "mylib"
    version = "1.0.0"
    settings = "os", "compiler", "build_type", "arch"
    exports_sources = "CMakeLists.txt", "src/*", "include/*"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["mylib"]
```

```bash
# 本地创建包
conan create . --build-folder=build

# 上传到Conan中心（需先登录）
conan remote add myremote https://my-server.com/artifactory/api/conan/conan
conan upload mylib/1.0.0 -r myremote --all
```

## 三、注意事项与常见陷阱

1. **Conan 2.x不兼容1.x**：Conan 2.0语法与1.x有较大差异，教程和文档要注意版本
2. **profile配置**：确保profile中的编译器版本与实际一致，否则可能链接失败
3. **lockfile**：生产项目应使用lockfile固定所有依赖的确切版本
4. **包缓存**：Conan在`~/.conan2`目录缓存已下载的包，清理可用`conan cache clean`
5. **二进制包**：Conan中心不一定有所有平台的预编译二进制，`--build=missing`会本地编译
