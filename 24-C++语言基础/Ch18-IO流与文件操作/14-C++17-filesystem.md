# C++17 filesystem

## 一、概念说明

`std::filesystem`（C++17, `<filesystem>`）提供跨平台的文件系统操作，包括路径处理、目录遍历、文件复制/移动/删除等。核心类型：
- `std::filesystem::path`：表示文件路径
- `directory_iterator`：遍历目录
- `recursive_directory_iterator`：递归遍历

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // 路径操作
    fs::path p("/home/user/documents/report.pdf");
    std::cout << "完整路径: " << p << std::endl;
    std::cout << "父路径: " << p.parent_path() << std::endl;
    std::cout << "文件名: " << p.filename() << std::endl;
    std::cout << "扩展名: " << p.extension() << std::endl;
    std::cout << "不含扩展名: " << p.stem() << std::endl;

    // 路径拼接
    fs::path dir = "/home/user";
    fs::path file = "test.txt";
    fs::path full = dir / file;
    std::cout << "拼接: " << full << std::endl;

    // 文件信息
    if (fs::exists(full)) {
        std::cout << "文件大小: " << fs::file_size(full) << " 字节" << std::endl;
        std::cout << "是否常规文件: " << fs::is_regular_file(full) << std::endl;
    }

    return 0;
}
```

**输出（假设文件存在）：**
```
完整路径: "/home/user/documents/report.pdf"
父路径: "/home/user/documents"
文件名: "report.pdf"
扩展名: ".pdf"
不含扩展名: "report"
拼接: "/home/user/test.txt"
文件大小: 1024 字节
是否常规文件: 1
```

## 二、具体用法

### 2.1 目录遍历

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // 遍历当前目录
    std::cout << "当前目录文件:" << std::endl;
    for (const auto& entry : fs::directory_iterator(".")) {
        if (entry.is_regular_file()) {
            std::cout << "  文件: " << entry.path().filename()
                      << " (" << entry.file_size() << " 字节)" << std::endl;
        } else if (entry.is_directory()) {
            std::cout << "  目录: " << entry.path().filename() << std::endl;
        }
    }

    // 递归遍历
    std::cout << "\n递归遍历（限深度3）:" << std::endl;
    for (const auto& entry : fs::recursive_directory_iterator(
             ".", fs::directory_options::skip_permission_denied)) {
        if (entry.depth() > 3) continue;
        std::cout << std::string(entry.depth() * 2, ' ')
                  << entry.path().filename() << std::endl;
    }

    return 0;
}
```

**输出（示例）：**
```
当前目录文件:
  文件: main.cpp (1234 字节)
  目录: src

递归遍历（限深度3）:
main.cpp
src
  main.cpp
  utils
    helper.cpp
```

### 2.2 文件操作

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    fs::create_directory("test_dir");
    std::cout << "创建目录: test_dir" << std::endl;

    // 创建文件
    std::ofstream("test_dir/file.txt") << "内容";

    // 复制
    fs::copy("test_dir/file.txt", "test_dir/copy.txt");
    std::cout << "复制完成" << std::endl;

    // 重命名/移动
    fs::rename("test_dir/copy.txt", "test_dir/moved.txt");
    std::cout << "移动完成" << std::endl;

    // 删除
    fs::remove("test_dir/moved.txt");
    fs::remove_all("test_dir"); // 递归删除
    std::cout << "删除完成" << std::endl;

    return 0;
}
```

**输出：**
```
创建目录: test_dir
复制完成
移动完成
删除完成
```

## 三、注意事项与常见陷阱

- **需要链接`-lstdc++fs`**（旧版GCC）：GCC 9+一般不需要。
- **路径分隔符`/`跨平台**：`path`自动处理。
- **`directory_iterator`不排序**：需要排序需收集后`sort`。
- **`file_size`对目录无效**：只对常规文件有效。
- **符号链接需要`symlink_status`检测**：`status`会跟踪链接目标。
- **异常版本和错误码版本都有**：`fs::exists(p)` vs `fs::exists(p, ec)`。
