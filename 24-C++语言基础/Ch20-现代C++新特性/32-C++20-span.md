# C++20 span

## 一、概念说明

`std::span`（`<span>`）是连续对象序列的非拥有视图。它结合了指针+长度的安全性，替代`T*`+`size_t`参数对。类似`string_view`但适用于任意类型。

```cpp
#include <iostream>
#include <span>
#include <vector>
#include <array>

// 用span代替指针+长度
void print(std::span<const int> data) {
    std::cout << "大小=" << data.size() << ": ";
    for (int x : data) std::cout << x << " ";
    std::cout << std::endl;
}

int main() {
    // 从不同来源创建span
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {6, 7, 8};
    std::array<int, 3> stdarr = {9, 10, 11};

    print(arr);       // 从数组
    print(vec);       // 从vector
    print(stdarr);    // 从array

    // 动态span
    std::span<int> dynamicSpan(arr);
    dynamicSpan[0] = 100;
    std::cout << "修改后arr[0]=" << arr[0] << std::endl;

    // 固定大小span
    std::span<int, 5> fixedSpan(arr);
    std::cout << "固定大小: " << fixedSpan.size() << std::endl;

    return 0;
}
```

**输出：**
```
大小=5: 1 2 3 4 5
大小=3: 6 7 8
大小=3: 9 10 11
修改后arr[0]=100
固定大小: 5
```

## 二、具体用法

### 2.1 span的操作

```cpp
#include <iostream>
#include <span>

int main() {
    int data[] = {10, 20, 30, 40, 50};
    std::span<int> sp(data);

    // 子视图
    auto sub = sp.subspan(1, 3); // [20, 30, 40]
    std::cout << "subspan(1,3): ";
    for (int x : sub) std::cout << x << " ";
    std::cout << std::endl;

    // first/last
    auto first2 = sp.first(2);
    auto last2 = sp.last(2);
    std::cout << "first(2): " << first2[0] << " " << first2[1] << std::endl;
    std::cout << "last(2): " << last2[0] << " " << last2[1] << std::endl;

    // data()获取裸指针
    int* ptr = sp.data();
    std::cout << "data(): " << *ptr << std::endl;

    return 0;
}
```

**输出：**
```
subspan(1,3): 20 30 40
first(2): 10 20
last(2): 40 50
data(): 10
```

## 三、注意事项与常见陷阱

- **`span`不拥有数据**：底层数据销毁后span悬垂。
- **`span<T>`可写，`span<const T>`只读**。
- **固定大小`span<T, N>`编译期检查大小**：更安全。
- **从`vector`创建span是O(1)**：不拷贝数据。
- **`span`替代`T* + size_t`参数对**：更安全、更简洁。
