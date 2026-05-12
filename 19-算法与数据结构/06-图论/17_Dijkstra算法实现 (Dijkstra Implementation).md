# Dijkstra 算法实现 (Dijkstra Implementation)

  使用最小堆（优先队列）优化，每次 O(logV) 取出最小距离顶点，总复杂度 O((V+E)logV)。


```javascript
class MinHeap {
  constructor() { this.heap = []; }
  push(v, dist) {
    this.heap.push({v, dist});
    this.bubbleUp(this.heap.length-1);
  }
  pop() {
    if (this.heap.length===1) return this.heap.pop();
    const top = this.heap[0];
    this.heap[0] = this.heap.pop();
    this.bubbleDown(0);
    return top;
  }
  bubbleUp(i) {
    while (i > 0) {
      const parent = Math.floor((i - 1) / 2);
      if (this.heap[i].dist >= this.heap[parent].dist) break;
      [this.heap[i], this.heap[parent]] = [this.heap[parent], this.heap[i]];
      i = parent;
    }
  }
  bubbleDown(i) {
    const n = this.heap.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.heap[l].dist < this.heap[smallest].dist) smallest = l;
      if (r < n && this.heap[r].dist < this.heap[smallest].dist) smallest = r;
      if (smallest === i) break;
      [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
      i = smallest;
    }
  }
  get size() { return this.heap.length; }
}
```

## 完整 Dijkstra（优先队列版）

```javascript
// 使用优先队列的完整 Dijkstra
class MinHeap {
  constructor() { this.heap = []; }
  push(v, dist) {
    this.heap.push({v, dist});
    this.bubbleUp(this.heap.length - 1);
  }
  pop() {
    if (this.heap.length === 1) return this.heap.pop();
    const top = this.heap[0];
    this.heap[0] = this.heap.pop();
    this.bubbleDown(0);
    return top;
  }
  bubbleUp(i) {
    while (i > 0) {
      const parent = Math.floor((i - 1) / 2);
      if (this.heap[i].dist >= this.heap[parent].dist) break;
      [this.heap[i], this.heap[parent]] = [this.heap[parent], this.heap[i]];
      i = parent;
    }
  }
  bubbleDown(i) {
    const n = this.heap.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.heap[l].dist < this.heap[smallest].dist) smallest = l;
      if (r < n && this.heap[r].dist < this.heap[smallest].dist) smallest = r;
      if (smallest === i) break;
      [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
      i = smallest;
    }
  }
  get size() { return this.heap.length; }
}

function dijkstra(graph, start) {
  const dist = {};
  const parent = {};
  for (const v in graph) { dist[v] = Infinity; parent[v] = -1; }
  dist[start] = 0;

  const pq = new MinHeap();
  pq.push(start, 0);

  while (pq.size > 0) {
    const { v: u, dist: d } = pq.pop();
    if (d > dist[u]) continue;  // 懒删除：跳过已更新的旧条目

    for (const [v, weight] of Object.entries(graph[u])) {
      if (dist[u] + weight < dist[v]) {
        dist[v] = dist[u] + weight;
        parent[v] = u;
        pq.push(v, dist[v]);
      }
    }
  }
  return { dist, parent };
}

// 还原路径
function getPath(parent, target) {
  const path = [];
  for (let v = target; v !== -1; v = parent[v]) path.push(v);
  return path.reverse();
}
```

```cpp
// C++ 优先队列 Dijkstra
#include <queue>
#include <vector>
#include <climits>
using namespace std;

void dijkstra(int src, vector<vector<pair<int,int>>>& adj, vector<int>& dist) {
    int n = adj.size();
    dist.assign(n, INT_MAX);
    dist[src] = 0;
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
}
```

## 复杂度对比

  | 实现方式 | 时间复杂度 | 适用场景 |
  | --- | --- | --- |
  | 数组扫描 | O(V^2) | 稠密图 |
  | 二叉堆 | O((V+E)logV) | 通用 |
  | 斐波那契堆 | O(E + VlogV) | 理论最优 |

## 常见陷阱

  - 忘记 `if (d > dist[u]) continue` 懒删除，导致重复处理
  - 使用 Array.shift() 而非优先队列，退化为 O(V^2)
  - 混淆 Dijkstra 和 BFS，Dijkstra 不适用于负权边

## 交互演示
