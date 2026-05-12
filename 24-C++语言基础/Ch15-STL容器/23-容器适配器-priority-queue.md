# 容器适配器：priority_queue

## 一、概念说明

`std::priority_queue`是按优先级出队的容器适配器（C++标准 §23.6.6.2），默认是大顶堆（最大元素先出队）。底层基于`vector`和堆算法（`make_heap`、`push_heap`、`pop_heap`）。priority_queue常用于Top-K问题、Dijkstra算法、任务调度等场景。

### 1.1 堆的基本性质

```
大顶堆：每个节点 >= 其子节点，根节点最大
小顶堆：每个节点 <= 其子节点，根节点最小

底层存储：vector（完全二叉树）
- 父节点：(i-1)/2
- 左子节点：2*i+1
- 右子节点：2*i+2
```

## 二、具体用法

### 2.1 基本用法

```cpp
#include <queue>
#include <iostream>
#include <vector>

int main() {
    // 大顶堆（默认）
    std::priority_queue<int> max_pq;
    max_pq.push(3);
    max_pq.push(1);
    max_pq.push(4);
    max_pq.push(1);
    max_pq.push(5);

    std::cout << "top: " << max_pq.top() << std::endl;  // 5（最大）
    max_pq.pop();  // 移除5
    std::cout << "top: " << max_pq.top() << std::endl;  // 4

    // 从范围构造
    std::vector<int> data = {3, 1, 4, 1, 5, 9};
    std::priority_queue<int> pq(data.begin(), data.end());
}
```

### 2.2 小顶堆

```cpp
#include <functional>

void min_heap() {
    // 小顶堆：第三个模板参数用greater
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
    min_pq.push(3);
    min_pq.push(1);
    min_pq.push(4);

    std::cout << "top: " << min_pq.top() << std::endl;  // 1（最小）
}
```

### 2.3 自定义比较

```cpp
struct Task {
    int priority;
    std::string name;

    // 大顶堆：priority大的先出
    bool operator<(const Task& other) const {
        return priority < other.priority;
    }
};

void custom_compare() {
    std::priority_queue<Task> tasks;
    tasks.push({1, "低优先级"});
    tasks.push({3, "高优先级"});
    tasks.push({2, "中优先级"});

    while (!tasks.empty()) {
        std::cout << tasks.top().name << std::endl;
        tasks.pop();
    }
    // 高优先级、中优先级、低优先级
}

// 使用lambda（C++20）
void lambda_compare() {
    auto cmp = [](const Task& a, const Task& b) {
        return a.priority > b.priority;  // 小顶堆
    };
    std::priority_queue<Task, std::vector<Task>, decltype(cmp)> pq(cmp);
}
```

### 2.4 实用示例

```cpp
#include <vector>

// Top-K问题：找前K大的元素
std::vector<int> top_k(const std::vector<int>& nums, int k) {
    // 小顶堆
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;

    for (int num : nums) {
        min_pq.push(num);
        if ((int)min_pq.size() > k)
            min_pq.pop();  // 弹出最小的
    }

    std::vector<int> result;
    while (!min_pq.empty()) {
        result.push_back(min_pq.top());
        min_pq.pop();
    }
    return result;
}

// 合并K个有序链表（简化版）
#include <tuple>
void merge_k_sorted() {
    std::vector<std::vector<int>> lists = {{1,4,7}, {2,5,8}, {3,6,9}};
    // (值, 列表索引, 元素索引)
    std::priority_queue<
        std::tuple<int,int,int>,
        std::vector<std::tuple<int,int,int>>,
        std::greater<>
    > pq;

    for (size_t i = 0; i < lists.size(); ++i)
        if (!lists[i].empty())
            pq.push({lists[i][0], i, 0});

    while (!pq.empty()) {
        auto [val, li, ei] = pq.top();
        pq.pop();
        std::cout << val << " ";
        if (ei + 1 < (int)lists[li].size())
            pq.push({lists[li][ei+1], li, ei+1});
    }
}
```

## 三、注意事项与常见陷阱

1. **默认大顶堆**：`std::greater<>`实现小顶堆（注意方向）
2. **不能遍历，只能访问堆顶**：没有迭代器
3. **`pop()`不返回值**：需要先`top()`获取
4. **底层是vector**：连续内存，缓存友好
5. **应用**：Dijkstra算法、Top-K问题、任务调度、合并有序序列
6. **`emplace`（C++11）**：直接构造元素
7. **自定义类型需要`operator<`或自定义比较器**
