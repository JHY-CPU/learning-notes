# 投影与ranges（C++20）

## 一、概念说明

C++20 Ranges引入了**投影**（Projection）概念，允许算法在比较前先对元素进行变换。同时提供了视图管道（View Pipeline），实现惰性数据处理。

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

    // 投影：按age排序
    std::ranges::sort(people, std::ranges::less{}, &Person::age);
    // people按age排序：Bob(25), Alice(30), Charlie(35)

    // 投影：按name查找
    auto it = std::ranges::find(people, "Alice", &Person::name);

    // 投影：最大age
    auto oldest = std::ranges::max_element(people, {}, &Person::age);
    std::cout << oldest->name << std::endl;  // Charlie
}
```

### 2.2 管道语法

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

## 三、注意事项

- 投影避免了手动提取字段的繁琐
- 视图是惰性的，不立即计算
- ranges算法在`std::ranges`命名空间
- 视图可以组合成管道
