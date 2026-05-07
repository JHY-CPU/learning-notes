# 容器适配器：queue

## 一、概念说明

`std::queue`是FIFO（先进先出）容器适配器，只允许在一端插入、另一端删除。默认基于`deque`。

## 二、具体用法

```cpp
#include <queue>
#include <iostream>
#include <list>

int main() {
    std::queue<int> q;

    // 操作
    q.push(1);
    q.push(2);
    q.push(3);

    std::cout << "front: " << q.front() << std::endl;  // 1
    std::cout << "back: " << q.back() << std::endl;    // 3

    q.pop();  // 移除1
    std::cout << "front: " << q.front() << std::endl;  // 2

    // 基于list
    std::queue<int, std::list<int>> list_q;

    // 遍历
    while (!q.empty()) {
        std::cout << q.front() << " ";
        q.pop();
    }
    // 2 3
}
```

## 三、注意事项

- 不支持随机访问和迭代器
- 底层容器需支持`front`、`back`、`push_back`、`pop_front`
- 应用：BFS、任务队列、消息队列
- `priority_queue`是不同概念（按优先级出队）
