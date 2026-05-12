# C++17 filesystem

## 一、概念说明

`std::filesystem`（C++17 §29，`<filesystem>`头文件）提供跨平台的文件系统操作：路径处理、目录遍历、文件操作、属性查询等。基于Boost.Filesystem设计，统一了不同操作系统的文件系统API。

### 1.1 核心组件

| 组件 | 用途 |
|------|------|
| `path` | 路径表示和操作 |
| `directory_entry` | 目录项（文件/目录） |
| `directory_iterator` | 非递归目录遍历 |
| `recursive_directory_iterator` | 递归目录遍历 |
| 函数 | `exists`, `create_directories`, `copy`, `remove`等 |

```cpp
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main() {
    // 路径操作
    fs::path p("/home/user/docs/file.txt");
    std::cout << "根路径: " << p.root_path() << std::endl;
    std::cout << "父路径: " << p.parent_path() << std::endl;
    std::cout << "文件名: " << p.filename() << std::endl;
    std::cout << "扩展名: " << p.extension() << std::endl;
    std::cout << "主干名: " << p.stem() << std::endl;

    // 路径拼接
    fs::path dir = "output";
    fs::path full = dir / "sub" / "file.dat";
    std::cout << "拼接: " << full << std::endl;

    // 文件操作
    fs::create_directories("test/sub");
    std::ofstream("test/file.txt") << "Hello FS";
    std::cout << "文件存在: " << fs::exists("test/file.txt") << std::endl;
    std::cout << "文件大小: " << fs::file_size("test/file.txt") << " 字节" << std::endl;

    // 清理
    fs::remove_all("test");

    return 0;
}
```

**输出：**
```
根路径: "/"
父路径: "/home/user/docs"
文件名: "file.txt"
扩展名: ".txt"
主干名: "file"
拼接: "output/sub/file.dat"
文件存在: 1
文件大小: 9 字节
```

## 二、具体用法

### 2.1 目录遍历

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // 非递归遍历
    std::cout << "=== 当前目录 ===" << std::endl;
    for (const auto& entry : fs::directory_iterator(".")) {
        if (entry.is_regular_file()) {
            std::cout << entry.path().filename()
                      << " (" << entry.file_size() << " bytes)" << std::endl;
        } else if (entry.is_directory()) {
            std::cout << "[" << entry.path().filename() << "]" << std::endl;
        }
    }

    // 递归遍历
    std::cout << "\n=== 递归遍历 ===" << std::endl;
    for (const auto& entry : fs::recursive_directory_iterator(".")) {
        if (entry.is_regular_file() && entry.path().extension() == ".md") {
            std::cout << entry.path() << std::endl;
        }
    }

    return 0;
}
```

### 2.2 文件操作

```cpp
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main() {
    // 创建目录
    fs::create_directories("a/b/c");

    // 写文件
    std::ofstream("a/test.txt") << "Hello";
    std::ofstream("a/b/test.txt") << "World";

    // 复制
    fs::copy("a/test.txt", "a/test_copy.txt");

    // 重命名
    fs::rename("a/test_copy.txt", "a/renamed.txt");

    // 检查
    std::cout << "a/test.txt 存在: " << fs::exists("a/test.txt") << std::endl;
    std::cout << "a/renamed.txt 存在: " << fs::exists("a/renamed.txt") << std::endl;

    // 空间信息
    auto space = fs::space(".");
    std::cout << "可用空间: " << space.available / (1024*1024) << " MB" << std::endl;

    // 临时目录
    std::cout << "临时目录: " << fs::temp_directory_path() << std::endl;

    // 清理
    fs::remove_all("a");

    return 0;
}
```

### 2.3 路径操作

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    fs::path p = "/home/user/documents/report.pdf";

    // 路径分解
    std::cout << "完整路径: " << p << std::endl;
    std::cout << "根名: " << p.root_name() << std::endl;
    std::cout << "根路径: " << p.root_path() << std::endl;
    std::cout << "相对路径: " << p.relative_path() << std::endl;
    std::cout << "父路径: " << p.parent_path() << std::endl;
    std::cout << "文件名: " << p.filename() << std::endl;
    std::cout << "主干名: " << p.stem() << std::endl;
    std::cout << "扩展名: " << p.extension() << std::endl;

    // 路径修改
    p.replace_extension(".html");
    std::cout << "替换扩展名: " << p << std::endl;

    // 相对路径
    fs::path base = "/home/user";
    fs::path full = "/home/user/documents/report.pdf";
    std::cout << "相对: " << fs::relative(full, base) << std::endl;

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **旧版GCC需要链接`-lstdc++fs`**：GCC 9+一般不需要。
2. **路径分隔符`/`跨平台兼容**：`path`类自动处理不同系统的分隔符。
3. **`directory_iterator`不排序**：需要排序要先收集再排序。
4. **`file_size`对目录无效**：只适用于常规文件。
5. **路径比较是字典序的**：不同系统可能有不同结果。
6. **操作可能抛`filesystem_error`异常**：建议使用异常处理。
7. **详细内容参见Ch18 IO流章节**。
