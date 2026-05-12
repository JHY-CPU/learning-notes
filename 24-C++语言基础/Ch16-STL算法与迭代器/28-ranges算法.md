# ranges算法

## 一、概念说明

`std::ranges`命名空间（C++标准 §25.3-§25.4）提供了算法的Ranges版本，直接接受范围（而非迭代器对），支持投影，并返回迭代器+哨位对。这使得算法调用更简洁，投影功能消除了大量样板代码。

### 1.1 与传统算法的对比

| 特性 | 传统算法 | ranges算法 |
|------|---------|-----------|
| 参数 | 迭代器对 | 范围 |
| 投影 | 不支持 | 支持 |
| 返回值 | 迭代器 | 迭代器+哨位 |
| 管道 | 不支持 | 支持 |

## 二、具体用法

```cpp
#include <ranges>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>

struct Person {
    std::string name;
    int age;
};

int main() {
    std::vector<int> v = {5, 3, 1, 4, 2};

    // 直接接受范围
    std::ranges::sort(v);

    // ranges::find
    auto it = std::ranges::find(v, 3);
    if (it != v.end()) std::cout << "找到: " << *it << std::endl;

    // ranges::for_each
    std::ranges::for_each(v, [](int x) { std::cout << x << " "; });

    // ranges::all_of / any_of / none_of
    bool all_positive = std::ranges::all_of(v, [](int x) { return x > 0; });

    // ranges::copy
    std::vector<int> dst(v.size());
    std::ranges::copy(v, dst.begin());

    // ranges::count
    auto cnt = std::ranges::count(v, 3);

    // 投影
    std::vector<Person> people = {{"Alice", 30}, {"Bob", 25}};
    std::ranges::sort(people, {}, &Person::age);  // 按age排序
    auto found = std::ranges::find(people, "Alice", &Person::name);
}
```

## 三、注意事项

1. **ranges算法直接接受范围**：更简洁
2. **支持投影**：无需手动提取字段
3. **返回迭代器+哨位对**：而非单个迭代器
4. **与传统算法并存**：可以混用
5. **需要C++20支持**

## 四、管道语法详解

```cpp
#include <ranges>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 管道语法：过滤 + 变换
    auto result = v
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; })
        | std::views::take(3);

    for (int x : result) std::cout << x << " ";  // 4 16 36

    // 惰性求值：不复制数据，按需计算
    auto evens = v | std::views::filter([](int x) { return x % 2 == 0; });
    // 此时还没有遍历，访问时才计算

    // 视图组合
    auto view = v
        | std::views::drop(5)    // 跳过前 5 个
        | std::views::reverse    // 反转
        | std::views::take(3);   // 取前 3 个
    // 结果: {10, 9, 8}
}
```

## 五、常用视图

| 视图 | 功能 | 示例 |
| --- | --- | --- |
| `filter` | 过滤元素 | `v \| views::filter(pred)` |
| `transform` | 变换每个元素 | `v \| views::transform(f)` |
| `take` | 取前 n 个 | `v \| views::take(n)` |
| `drop` | 跳过前 n 个 | `v \| views::drop(n)` |
| `reverse` | 反转视图 | `v \| views::reverse` |
| `join` | 展平嵌套范围 | `nested \| views::join` |
| `split` | 按分隔符拆分 | `str \| views::split(' ')` |
| `iota` | 生成整数序列 | `views::iota(0, 10)` |
| `keys` / `values` | map 的键/值 | `m \| views::keys` |

## 六、投影（Projection）详解

```cpp
struct Employee {
    std::string name;
    int age;
    double salary;
};

int main() {
    std::vector<Employee> staff = {
        {"Alice", 30, 8000}, {"Bob", 25, 6000}, {"Charlie", 35, 9000}
    };

    // 按薪资排序（投影自动提取字段）
    std::ranges::sort(staff, std::less<>{}, &Employee::salary);

    // 按年龄查找
    auto it = std::ranges::find(staff, 30, &Employee::age);

    // 找薪资最高的
    auto [maxIt] = std::ranges::max_element(staff, {}, &Employee::salary);

    // 变换 + 投影组合
    std::vector<std::string> names;
    std::ranges::transform(staff, std::back_inserter(names),
        [](const Employee& e) { return e.name; });
}
```

## 七、与传统算法的迁移指南

```cpp
// 传统写法
std::sort(v.begin(), v.end());
auto it = std::find(v.begin(), v.end(), 42);
std::copy(v.begin(), v.end(), std::back_inserter(dst));

// ranges 写法
std::ranges::sort(v);
auto it = std::ranges::find(v, 42);
std::ranges::copy(v, std::back_inserter(dst));

// ranges 独有优势
auto sorted = v | std::views::sort;  // 不修改原数组！
auto filtered = v | std::views::filter([](int x) { return x > 0; });
```

## 八、注意事项补充

1. **惰性求值**：视图不立即计算，遍历时才执行，注意生命周期
2. **视图不拥有数据**：原始数据销毁后视图失效
3. **C++20 需要较新的编译器**：GCC 10+, Clang 10+, MSVC 19.29+
4. **性能**：ranges 与传统算法在开启优化后性能相当
5. **调试**：视图的惰性求值可能使调试更困难
