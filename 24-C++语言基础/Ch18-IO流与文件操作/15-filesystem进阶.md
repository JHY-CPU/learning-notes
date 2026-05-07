# filesystem进阶

## 一、概念说明

filesystem进阶操作包括：文件属性查询、权限管理、时间戳操作、临时目录、符号链接、硬链接、空间信息等。这些功能使得C++可以替代许多shell脚本中的文件操作。

```cpp
#include <iostream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

int main() {
    fs::path p = "test.txt";
    std::ofstream(p) << "测试内容";

    // 文件属性
    auto status = fs::status(p);
    std::cout << "文件类型: " << static_cast<int>(status.type()) << std::endl;
    std::cout << "权限: " << static_cast<int>(status.permissions()) << std::endl;

    // 文件大小
    std::cout << "大小: " << fs::file_size(p) << " 字节" << std::endl;

    // 最后修改时间
    auto ftime = fs::last_write_time(p);
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);
    std::cout << "修改时间: " << std::asctime(std::localtime(&cftime));

    // 磁盘空间
    auto space = fs::space(".");
    std::cout << "总空间: " << space.capacity / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "可用空间: " << space.available / (1024*1024*1024) << " GB" << std::endl;

    fs::remove(p);
    return 0;
}
```

**输出（示例）：**
```
文件类型: 1
权限: 438
大小: 12 字节
修改时间: Tue May  6 10:30:00 2026
总空间: 500 GB
可用空间: 200 GB
```

## 二、具体用法

### 2.1 文件查找

```cpp
#include <iostream>
#include <filesystem>
#include <vector>
#include <regex>

namespace fs = std::filesystem;

// 按扩展名查找
std::vector<fs::path> findFiles(const fs::path& dir,
                                 const std::string& ext) {
    std::vector<fs::path> results;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ext) {
            results.push_back(entry.path());
        }
    }
    return results;
}

// 按正则匹配查找
std::vector<fs::path> findFilesRegex(const fs::path& dir,
                                      const std::regex& pattern) {
    std::vector<fs::path> results;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string name = entry.path().filename().string();
            if (std::regex_match(name, pattern)) {
                results.push_back(entry.path());
            }
        }
    }
    return results;
}

int main() {
    // 查找所有.cpp文件
    auto cppFiles = findFiles(".", ".cpp");
    std::cout << "找到 " << cppFiles.size() << " 个.cpp文件" << std::endl;

    // 查找匹配模式的文件
    auto mdFiles = findFilesRegex(".", std::regex(R"(.*\.md)"));
    std::cout << "找到 " << mdFiles.size() << " 个.md文件" << std::endl;

    return 0;
}
```

**输出（示例）：**
```
找到 5 个.cpp文件
找到 10 个.md文件
```

### 2.2 目录同步

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void syncDirectory(const fs::path& src, const fs::path& dst) {
    fs::create_directories(dst);

    for (const auto& entry : fs::recursive_directory_iterator(src)) {
        auto relative = fs::relative(entry.path(), src);
        auto target = dst / relative;

        if (entry.is_directory()) {
            fs::create_directories(target);
        } else if (entry.is_regular_file()) {
            // 仅复制修改时间更新的文件
            if (!fs::exists(target) ||
                fs::last_write_time(entry.path()) > fs::last_write_time(target)) {
                fs::copy_file(entry.path(), target,
                              fs::copy_options::overwrite_existing);
                std::cout << "复制: " << relative << std::endl;
            }
        }
    }
}

int main() {
    fs::create_directories("src_dir/sub");
    std::ofstream("src_dir/a.txt") << "a";
    std::ofstream("src_dir/sub/b.txt") << "b";

    syncDirectory("src_dir", "dst_dir");
    std::cout << "同步完成" << std::endl;

    fs::remove_all("src_dir");
    fs::remove_all("dst_dir");
    return 0;
}
```

**输出：**
```
复制: a.txt
复制: sub\b.txt
同步完成
```

## 三、注意事项与常见陷阱

- **`file_size`对目录和符号链接行为不同**：目录无大小，符号链接指向目标大小。
- **`last_write_time`的时区处理复杂**：需要转换为`system_clock`。
- **`copy_options`控制复制行为**：`overwrite_existing`、`recursive`等。
- **`create_directories`可创建多级目录**：类似`mkdir -p`。
- **`equivalent`检查两路径是否指向同一文件**：考虑符号链接。
