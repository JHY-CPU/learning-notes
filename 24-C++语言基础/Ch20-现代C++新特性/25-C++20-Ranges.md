# C++20 Ranges

## 一、概念说明

Ranges库（C++20 §24，`<ranges>`）提供函数式风格的数据处理管道。使用视图（views）和适配器（adapters）组合操作，惰性求值，避免不必要的拷贝。Ranges是C++20四大特性之一。

### 1.1 Ranges的三大组件

| 组件 | 功能 |
|------|------|
| 范围（Range） | 有begin/end的类型 |
| 视图（View） | 惰性转换管道 |
| 算法（Algorithm） | 支持投影的算法 |

```cpp
#include <iostream>
#include <ranges>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 管道操作：过滤偶数并平方
    auto result = nums
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; });

    std::cout << "偶数平方: ";
    for (int x : result) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // take / drop
    std::cout << "前3个: ";
    for (int x : nums | std::views::take(3))
        std::cout << x << " ";
    std::cout << std::endl;

    // reverse
    std::cout << "反转: ";
    for (int x : nums | std::views::reverse | std::views::take(5))
        std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
```

**输出：**
```
偶数平方: 4 16 36 64 100
前3个: 1 2 3
反转: 10 9 8 7 6
```

## 二、具体用法

### 2.1 常用视图

| 视图 | 作用 |
|------|------|
| `filter(pred)` | 过滤满足谓词的元素 |
| `transform(f)` | 对每个元素应用函数 |
| `take(n)` | 取前n个元素 |
| `drop(n)` | 跳过前n个元素 |
| `reverse` | 反转 |
| `join` | 展平嵌套范围 |
| `split(pattern)` | 按模式分割 |
| `iota(start, end)` | 生成连续整数 |

```cpp
#include <iostream>
#include <ranges>
#include <string>

int main() {
    // iota: 生成序列
    std::cout << "iota(1,6): ";
    for (int x : std::views::iota(1, 6))
        std::cout << x << " ";
    std::cout << std::endl;

    // split: 字符串分割
    std::string text = "one,two,three";
    std::cout << "split: ";
    for (auto word : text | std::views::split(',')) {
        std::string_view sv(word.begin(), word.end());
        std::cout << "'" << sv << "' ";
    }
    std::cout << std::endl;

    // join: 展平嵌套
    std::vector<std::vector<int>> nested = {{1, 2}, {3, 4}, {5, 6}};
    std::cout << "join: ";
    for (int x : nested | std::views::join)
        std::cout << x << " ";
    std::cout << std::endl;

    // 组合管道
    auto pipeline = std::views::iota(1)
                  | std::views::filter([](int x){ return x % 3 == 0; })
                  | std::views::transform([](int x){ return x * x; })
                  | std::views::take(5);

    std::cout << "3的倍数的平方前5个: ";
    for (int x : pipeline) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
```

**输出：**
```
iota(1,6): 1 2 3 4 5
split: 'one' 'two' 'three'
join: 1 2 3 4 5 6
3的倍数的平方前5个: 9 36 81 144 225
```

### 2.2 ranges算法

```cpp
#include <iostream>
#include <ranges>
#include <algorithm>
#include <vector>

struct Person {
    std::string name;
    int age;
};

int main() {
    std::vector<int> v = {5, 3, 1, 4, 2};

    // ranges::sort直接接受范围
    std::ranges::sort(v);

    // ranges::find
    auto it = std::ranges::find(v, 3);
    if (it != v.end()) std::cout << "找到: " << *it << std::endl;

    // 投影（Projection）
    std::vector<Person> people = {{"Alice", 30}, {"Bob", 25}, {"Charlie", 35}};
    std::ranges::sort(people, {}, &Person::age);  // 按age排序

    std::cout << "按年龄排序: ";
    for (const auto& p : people)
        std::cout << p.name << "(" << p.age << ") ";
    std::cout << std::endl;

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **视图是惰性的**：不执行计算，直到遍历时才求值。
2. **视图不拥有数据**：底层范围销毁后视图无效（悬垂引用）。
3. **`views::`是命名空间别名**：`std::views`等于`std::ranges::views`。
4. **需要C++20支持**：GCC 10+、Clang 13+。
5. **`std::ranges::sort`等算法支持投影**：不需要自定义比较器。
6. **视图的拷贝很轻量**：只拷贝迭代器和谓词的引用。
7. **详细内容参见Ch16 Ranges相关章节**。
