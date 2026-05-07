# 容器适配器：priority_queue

## 一、概念说明

`std::priority_queue`是按优先级出队的容器适配器，默认是大顶堆（最大元素先出队）。底层基于`vector`和`make_heap`。

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

    std::cout << "top: " << max_pq.top() << std::endl;  // 5
    max_pq.pop();
    std::cout << "top: " << max_pq.top() << std::endl;  // 4
}
```

### 2.2 小顶堆

```cpp
#include <functional>

void min_heap() {
    // 小顶堆
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
    min_pq.push(3);
    min_pq.push(1);
    min_pq.push(4);

    std::cout << "top: " << min_pq.top() << std::endl;  // 1
}
```

### 2.3 自定义比较

```cpp
struct Task {
    int priority;
    std::string name;
    bool operator<(const Task& other) const {
        return priority < other.priority;  // 大顶堆
    }
};

void custom_compare() {
    std::priority_queue<Task> tasks;
    tasks.push({1, "低优先级"});
    tasks.push({3, "高优先级"});
    tasks.push({2, "中优先级"});

    // 高优先级先出队
    while (!tasks.empty()) {
        std::cout << tasks.top().name << std::endl;
        tasks.pop();
    }
}
```

## 三、注意事项

- 默认大顶堆，`std::greater<>`实现小顶堆
- 不能遍历，只能访问堆顶
- `pop()`不返回值
- 应用：Dijkstra算法、Top-K问题、任务调度
