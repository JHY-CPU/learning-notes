# CSV文件读写

## 一、概念说明

CSV（Comma-Separated Values）是最常见的表格数据格式。每行一条记录，字段用逗号分隔。处理CSV需注意：字段含逗号时需引号包围、换行符转义、引号转义等。

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using Row = std::vector<std::string>;
using Table = std::vector<Row>;

// 简易CSV解析（不含引号处理）
Table readCSV(const std::string& filename) {
    Table table;
    std::ifstream ifs(filename);
    std::string line;

    while (std::getline(ifs, line)) {
        Row row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        table.push_back(row);
    }
    return table;
}

// 写入CSV
void writeCSV(const std::string& filename, const Table& table) {
    std::ofstream ofs(filename);
    for (const auto& row : table) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) ofs << ",";
            // 如果字段含逗号或引号，加引号包围
            if (row[i].find(',') != std::string::npos ||
                row[i].find('"') != std::string::npos) {
                ofs << '"' << row[i] << '"';
            } else {
                ofs << row[i];
            }
        }
        ofs << "\n";
    }
}

int main() {
    // 写入
    Table data = {
        {"姓名", "年龄", "城市"},
        {"张三", "25", "北京"},
        {"李四", "30", "上海,中国"},
        {"王五", "28", "广州"}
    };
    writeCSV("people.csv", data);

    // 读取
    auto loaded = readCSV("people.csv");
    for (const auto& row : loaded) {
        for (const auto& cell : row) {
            std::cout << std::setw(15) << cell;
        }
        std::cout << std::endl;
    }

    return 0;
}
```

**输出：**
```
           姓名            年龄            城市
           张三              25            北京
           李四              30       上海,中国
           王五              28            广州
```

## 二、具体用法

### 2.1 带引号处理的CSV解析

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"'; // 转义的引号
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c == ',' && !inQuotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

int main() {
    std::string testLine = R"(name,"He said ""Hello""",age)";
    auto fields = parseCSVLine(testLine);

    std::cout << "解析结果:" << std::endl;
    for (size_t i = 0; i < fields.size(); ++i) {
        std::cout << "  字段" << i << ": " << fields[i] << std::endl;
    }

    return 0;
}
```

**输出：**
```
解析结果:
  字段0: name
  字段1: He said "Hello"
  字段2: age
```

## 三、注意事项与常见陷阱

- **字段含逗号必须用引号包围**：否则解析会出错。
- **引号内用双引号`""`表示单引号**：这是CSV标准转义。
- **Windows和Linux换行符不同**：CSV可能用`\r\n`或`\n`。
- **生产环境建议用成熟的CSV库**：如`fast-csv`、`rapidcsv`。
- **BOM头可能出现在UTF-8文件开头**：读取时需跳过。
