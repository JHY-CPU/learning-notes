# 深度优先搜索（DFS）

## 1. DFS基本思想

深度优先搜索（DFS）从起点出发，沿着一条路径尽可能深入，直到无法继续时回退到上一个分叉点，尝试其他路径。DFS可以用**递归**或**显式栈**实现。

### DFS框架

```c
void dfs(当前状态) {
    if (到达目标状态) {
        记录/输出结果;
        return;
    }
    if (当前状态不合法) return;  // 剪枝
    标记当前状态为已访问;
    for (所有邻接/可达状态) {
        dfs(下一状态);
    }
    恢复当前状态（可选）;  // 回溯
}
```

## 2. DFS实现（邻接表）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAXN 10005

int head[MAXN], to[MAXN], nxt[MAXN], edge_cnt = 0;
bool visited[MAXN];

void addEdge(int u, int v) {
    to[edge_cnt] = v;
    nxt[edge_cnt] = head[u];
    head[u] = edge_cnt++;
}

void dfs(int u) {
    visited[u] = true;
    printf("访问节点: %d\n", u);
    for (int i = head[u]; i != -1; i = nxt[i]) {
        int v = to[i];
        if (!visited[v]) dfs(v);
    }
}
```

## 3. 连通分量

求无向图的连通分量个数及每个分量的节点。

```c
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#define MAXN 10005

int graph[MAXN][MAXN], n;
bool visited[MAXN];
int component[MAXN], compSize[MAXN], compCount = 0;

void dfsComponent(int u, int compId) {
    visited[u] = true;
    component[u] = compId;
    compSize[compId]++;
    for (int v = 0; v < n; v++) {
        if (graph[u][v] && !visited[v])
            dfsComponent(v, compId);
    }
}

void findConnectedComponents() {
    memset(visited, false, sizeof(visited));
    compCount = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            compSize[compCount] = 0;
            dfsComponent(i, compCount);
            compCount++;
        }
    }
    printf("共%d个连通分量\n", compCount);
}
```

## 4. 拓扑排序（DFS版）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAXN 10005

int head[MAXN], to[MAXN], nxt[MAXN], edge_cnt = 0;
int stack[MAXN], top = -1;
int visited[MAXN];  // 0: 未访问, 1: 正在访问, 2: 已完成
bool hasCycle = false;

void addEdge(int u, int v) {
    to[edge_cnt] = v;
    nxt[edge_cnt] = head[u];
    head[u] = edge_cnt++;
}

void topoDfs(int u) {
    visited[u] = 1;  // 标记为正在访问
    for (int i = head[u]; i != -1; i = nxt[i]) {
        int v = to[i];
        if (visited[v] == 1) { hasCycle = true; return; }  // 发现环
        if (!visited[v]) topoDfs(v);
    }
    visited[u] = 2;  // 标记为已完成
    stack[++top] = u;  // 入栈
}

void topologicalSort(int n) {
    memset(visited, 0, sizeof(visited));
    for (int i = 0; i < n; i++)
        if (!visited[i]) topoDfs(i);

    if (hasCycle) { printf("图中存在环！\n"); return; }
    printf("拓扑排序结果: ");
    while (top >= 0) printf("%d ", stack[top--]);
    printf("\n");
}
```

## 5. 判断有向图是否有环

```c
#include <stdio.h>
#include <stdbool.h>

#define MAXN 10005

int head[MAXN], to[MAXN], nxt[MAXN], edge_cnt = 0;
int color[MAXN];  // 0:白(未访问), 1:灰(正在访问), 2:黑(已完成)

bool dfsCycle(int u) {
    color[u] = 1;  // 灰色
    for (int i = head[u]; i != -1; i = nxt[i]) {
        int v = to[i];
        if (color[v] == 1) return true;   // 找到后向边，有环
        if (color[v] == 0 && dfsCycle(v)) return true;
    }
    color[u] = 2;  // 黑色
    return false;
}

bool hasCycleDirected(int n) {
    for (int i = 0; i < n; i++) color[i] = 0;
    for (int i = 0; i < n; i++)
        if (color[i] == 0 && dfsCycle(i)) return true;
    return false;
}
```

## 6. 基于栈的非递归DFS

```c
#include <stdio.h>
#include <stdbool.h>

#define MAXN 10005

int head[MAXN], to[MAXN], nxt[MAXN], edge_cnt = 0;
bool visited[MAXN];
int stack[MAXN], top;

void dfsIterative(int start) {
    top = 0;
    stack[top++] = start;
    visited[start] = true;

    while (top > 0) {
        int u = stack[--top];
        printf("访问节点: %d\n", u);
        for (int i = head[u]; i != -1; i = nxt[i]) {
            int v = to[i];
            if (!visited[v]) {
                visited[v] = true;
                stack[top++] = v;
            }
        }
    }
}
```

## 7. DFS找所有路径

```c
#include <stdio.h>
#include <stdbool.h>

#define MAXN 100

int graph[MAXN][MAXN], n;
int path[MAXN], pathLen;
bool visited[MAXN];

void findAllPaths(int u, int target) {
    path[pathLen++] = u;
    visited[u] = true;

    if (u == target) {
        printf("路径: ");
        for (int i = 0; i < pathLen; i++)
            printf("%d%s", path[i], i < pathLen - 1 ? " -> " : "\n");
    } else {
        for (int v = 0; v < n; v++) {
            if (graph[u][v] && !visited[v])
                findAllPaths(v, target);
        }
    }
    visited[u] = false;  // 回溯
    pathLen--;
}
```

## 8. DFS在网格中的应用

### 岛屿数量

```c
#include <stdio.h>

int grid[300][300], n, m;
int dx[] = {0, 0, 1, -1}, dy[] = {1, -1, 0, 0};

void dfsIsland(int x, int y) {
    if (x < 0 || x >= n || y < 0 || y >= m || grid[x][y] == 0)
        return;
    grid[x][y] = 0;  // 标记为已访问（沉岛）
    for (int d = 0; d < 4; d++)
        dfsIsland(x + dx[d], y + dy[d]);
}

int countIslands() {
    int count = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (grid[i][j] == 1) {
                count++;
                dfsIsland(i, j);
            }
    return count;
}
```
