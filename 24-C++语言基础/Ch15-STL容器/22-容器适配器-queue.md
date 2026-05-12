# 容器适配器：queue

## 一、概念说明

`std::queue`是FIFO（先进先出）容器适配器（C++标准 §23.6.6.1），只允许在一端插入（back）、另一端删除（front）。默认基于`deque`。queue不是独立的容器，而是对接口的限制，屏蔽了不需要的操作。

### 1.1 设计约束

```
queue的底层容器必须支持：
- front()：访问队头
- back()：访问队尾
- push_back()：尾部插入
- pop_front()：头部删除

因此只有deque和list满足要求，vector不支持pop_front。
```

## 二、具体用法

### 2.1 基本操作

```cpp
#include <queue>
#include <iostream>
#include <list>

int main() {
    std::queue<int> q;

    // 入队
    q.push(1);
    q.push(2);
    q.push(3);
    q.emplace(4);  // C++11

    // 访问
    std::cout << "front: " << q.front() << std::endl;  // 1（队头）
    std::cout << "back: " << q.back() << std::endl;    // 4（队尾）

    // 出队
    q.pop();  // 移除1
    std::cout << "front: " << q.front() << std::endl;  // 2

    // 大小
    std::cout << "size: " << q.size() << std::endl;
    std::cout << "empty: " << q.empty() << std::endl;

    // 基于list
    std::queue<int, std::list<int>> list_q;
}
```

### 2.2 性能分析

```cpp
/*
| 操作  | 时间复杂度 | 说明       |
|------|-----------|-----------|
| push | O(1)      | 队尾插入    |
| pop  | O(1)      | 队头移除    |
| front| O(1)      | 访问队头    |
| back | O(1)      | 访问队尾    |
| size | O(1)      | 取决于实现  |
*/
```

### 2.3 实用示例：BFS

```cpp
#include <queue>
#include <vector>
#include <iostream>

void bfs(int start, const std::vector<std::vector<int>>& graph) {
    std::queue<int> q;
    std::vector<bool> visited(graph.size(), false);

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        std::cout << "访问: " << node << std::endl;

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

// 任务调度
#include <string>
struct Task {
    int id;
    std::string name;
};

void task_scheduler() {
    std::queue<Task> tasks;
    tasks.push({1, "任务A"});
    tasks.push({2, "任务B"});
    tasks.push({3, "任务C"});

    while (!tasks.empty()) {
        auto task = tasks.front();
        tasks.pop();
        std::cout << "执行: " << task.name << std::endl;
    }
}
```

### 2.4 底层容器选择

```cpp
/*
| 底层容器 | 特性 | 适用场景 |
|---------|------|---------|
| deque   | 默认，两端操作快 | 大多数场景 |
| list    | 内存稳定，无预分配 | 需要稳定引用 |

注意：vector不能作为queue的底层容器
// std::queue<int, std::vector<int>> q;  // 编译错误
// vector不支持pop_front()
*/
```

## 三、注意事项与常见陷阱

1. **不支持随机访问和迭代器**：无法使用`std::sort`等算法
2. **底层容器需支持`front`、`back`、`push_back`、`pop_front`**：只有deque和list满足
3. **`priority_queue`是不同概念**：按优先级出队（大顶堆/小顶堆）
4. **queue没有clear方法**：清空需要循环pop，或重新构造
5. **`pop`不返回值**：与stack一样，需要先`front()`再`pop()`
6. **典型应用**：BFS、任务调度、生产者-消费者模型
