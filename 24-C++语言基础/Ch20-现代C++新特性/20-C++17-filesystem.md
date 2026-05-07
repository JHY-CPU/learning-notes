# C++17 filesystem

## 一、概念说明

`std::filesystem`（`<filesystem>`头文件）提供跨平台的文件系统操作：路径处理、目录遍历、文件操作、属性查询等。基于Boost.Filesystem。

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
    // 递归遍历当前目录
    for (const auto& entry : fs::recursive_directory_iterator(".")) {
        if (entry.is_regular_file()) {
            std::cout << entry.path().filename()
                      << " (" << entry.file_size() << " bytes)" << std::endl;
        }
    }

    // 临时目录
    std::cout << "临时目录: " << fs::temp_directory_path() << std::endl;

    // 空间信息
    auto space = fs::space(".");
    std::cout << "可用: " << space.available / (1024*1024) << " MB" << std::endl;

    return 0;
}
```

## 三、注意事项与常见陷阱

- **旧版GCC需要`-lstdc++fs`**：GCC 9+一般不需要。
- **路径分隔符`/`跨平台兼容**：`path`类自动处理。
- **`directory_iterator`不排序**。
- **`file_size`对目录无效**。
- **详细内容参见Ch18 IO流章节**。
