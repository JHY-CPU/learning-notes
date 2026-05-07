# format格式说明符

## 一、概念说明

`std::format`的格式说明符语法：`{[位置]:[填充][对齐][宽度][.精度][类型]}`

各部分含义：
- **位置**：参数索引 `{0}`, `{1}`
- **填充**：填充字符 `*`, `0`
- **对齐**：`<`左对齐, `>`右对齐, `^`居中
- **宽度**：最小宽度 `10`
- **精度**：浮点精度 `.2`
- **类型**：`d`整数, `f`定点, `e`科学计数, `x`十六进制

```cpp
#include <iostream>
#include <format>

int main() {
    // 宽度与对齐
    std::cout << std::format("|{:10}|", "left") << std::endl;     // 默认左对齐
    std::cout << std::format("|{:>10}|", "right") << std::endl;   // 右对齐
    std::cout << std::format("|{:^10}|", "center") << std::endl;  // 居中

    // 填充字符
    std::cout << std::format("|{:*<10}|", "pad") << std::endl;    // *填充左对齐
    std::cout << std::format("|{:0>10}|", 42) << std::endl;       // 0填充右对齐
    std::cout << std::format("|{:=^10}|", "mid") << std::endl;    // =居中

    // 精度
    std::cout << std::format("{:.2f}", 3.14159) << std::endl;     // 3.14
    std::cout << std::format("{:.4e}", 12345.0) << std::endl;     // 科学计数

    // 类型
    std::cout << std::format("{:d}", 255) << std::endl;           // 十进制
    std::cout << std::format("{:x}", 255) << std::endl;           // 小写十六进制
    std::cout << std::format("{:X}", 255) << std::endl;           // 大写十六进制
    std::cout << std::format("{:#x}", 255) << std::endl;          // 带前缀
    std::cout << std::format("{:o}", 255) << std::endl;           // 八进制
    std::cout << std::format("{:b}", 255) << std::endl;           // 二进制
    std::cout << std::format("{:#b}", 255) << std::endl;          // 带前缀二进制

    return 0;
}
```

**输出：**
```
|left      |
|     right|
|  center  |
|pad*******|
|0000000042|
|===mid===|
3.14
1.2345e+04
255
ff
FF
0xff
377
11111111
0b11111111
```

## 二、具体用法

### 2.1 组合格式说明符

```cpp
#include <iostream>
#include <format>

int main() {
    // 表格输出
    std::cout << std::format("{:<10} {:>8} {:>6}",
                             "Name", "Score", "Grade") << std::endl;
    std::cout << std::format("{:-<10} {:->8} {:->6}", "", "", "") << std::endl;
    std::cout << std::format("{:<10} {:>8} {:>6}",
                             "Alice", 95, "A") << std::endl;
    std::cout << std::format("{:<10} {:>8} {:>6}",
                             "Bob", 87, "B+") << std::endl;

    // 数字格式
    std::cout << std::format("\n千分位: {:L}", 1234567) << std::endl;
    std::cout << std::format("百分比: {:.1%}", 0.856) << std::endl;
    std::cout << std::format("正号: {:+d}", 42) << std::endl;
    std::cout << std::format("空格: {: d}", 42) << std::endl;

    return 0;
}
```

**输出：**
```
Name          Score   Grade
---------- -------- ------
Alice           95       A
Bob             87      B+

千分位: 1,234,567
百分比: 85.6%
正号: +42
空格:  42
```

### 2.2 动态宽度和精度

```cpp
#include <iostream>
#include <format>

int main() {
    // 使用参数作为宽度
    int width = 15;
    std::cout << std::format("|{:^{}}|", "dynamic", width) << std::endl;

    // 使用位置参数作为宽度
    std::cout << std::format("|{:^{1}}|", "test", 20) << std::endl;

    // 动态精度
    double val = 3.14159265;
    for (int prec = 1; prec <= 6; ++prec) {
        std::cout << std::format("精度{}: {:.{}f}", prec, val, prec) << std::endl;
    }

    return 0;
}
```

**输出：**
```
|    dynamic    |
|      test      |
精度1: 3.1
精度2: 3.14
精度3: 3.142
精度4: 3.1416
精度5: 3.14159
精度6: 3.141593
```

## 三、注意事项与常见陷阱

- **`L`修饰符使用locale的千位分隔符**：需要正确设置locale。
- **`%`类型是浮点百分比**：`0.856` 格式化为 `85.6%`。
- **`#`前缀对整数显示进制标记**：`0x`, `0b`, `0`。
- **精度对字符串是截断**：`{:.5}` 截断为前5个字符。
- **填充字符必须搭配对齐符号**：单独的填充字符无效。
