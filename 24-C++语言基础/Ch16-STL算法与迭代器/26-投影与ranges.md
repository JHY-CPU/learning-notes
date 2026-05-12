# 投影与ranges（C++20）

## 一、概念说明

C++20 Ranges（C++标准 §24.2-§26）引入了**投影**（Projection）概念，允许算法在比较前先对元素进行变换。同时提供了视图管道（View Pipeline），实现惰性数据处理。Ranges使STL算法更简洁、更强大。

### 1.1 Ranges的三大组件

| 组件 | 功能 |
|------|------|
| 范围（Range） | 有begin/end的类型 |
| 视图（View） | 惰性转换管道 |
| 算法（Algorithm） | 支持投影的算法 |

## 二、具体用法

### 2.1 投影

```cpp
#include <algorithm>
#include <ranges>
#include <vector>
#include <string>
#include <iostream>

struct Person {
    std::string name;
    int age;
};

int main() {
    std::vector<Person> people = {
        {"Alice", 30}, {"Bob", 25}, {"Charlie", 35}
    };

    // 投影：按age排序（不需要自定义比较器！）
    std::ranges::sort(people, std::ranges::less{}, &Person::age);

    // 投影：按name查找
    auto it = std::ranges::find(people, "Alice", &Person::name);

    // 投影：最大age
    auto oldest = std::ranges::max_element(people, {}, &Person::age);
    std::cout << oldest->name << std::endl;  // Charlie
}
```

### 2.2 视图管道

```cpp
void pipeline() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 视图管道：过滤+变换
    auto result = v
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; })
        | std::views::take(3);

    for (int x : result) std::cout << x << " ";
    // 4 16 36
}
```

### 2.3 ranges算法

```cpp
void ranges_algorithms() {
    std::vector<int> v = {5, 3, 1, 4, 2};

    // 直接接受范围（不需要begin/end）
    std::ranges::sort(v);

    auto it = std::ranges::find(v, 3);
    std::ranges::for_each(v, [](int x) { std::cout << x << " "; });

    bool all_pos = std::ranges::all_of(v, [](int x) { return x > 0; });
}
```

## 三、注意事项

1. **投影避免了手动提取字段的繁琐**
2. **视图是惰性的**：不立即计算
3. **ranges算法直接接受范围**：更简洁
4. **视图可以组合成管道**
5. **需要C++20编译器支持**
6. **视图不拥有数据**：原始容器必须保持有效
