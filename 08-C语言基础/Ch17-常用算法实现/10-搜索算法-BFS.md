# 广度优先搜索（BFS）

## 1. BFS基本思想

广度优先搜索（BFS）从起点开始，逐层向外扩展。先访问距离起点为1的所有节点，再访问距离为2的节点，依此类推。BFS使用**队列**作为核心数据结构。

### BFS框架

```c
void bfs(起点) {
    queue.push(起点);
    visited[起点] = true;
    while (!queue.empty()) {
        cur = queue.front(); queue.pop();
        for (cur的所有邻接点 next) {
            if (!visited[next]) {
                visited[next] = true;
                queue.push(next);
            }
        }
    }
}
```

## 2. BFS实现（邻接矩阵）

```c
#include <stdio.h>
#include <stdbool.h>

#define MAXN 1005

int graph[MAXN][MAXN], n;
bool visited[MAXN];
int queue[MAXN], front, rear;

void bfs(int start) {
    front = rear = 0;
    queue[rear++] = start;
    visited[start] = true;

    while (front < rear) {
        int cur = queue[front++];
        printf("访问节点: %d\n", cur);
        for (int i = 0; i < n; i++) {
            if (graph[cur][i] && !visited[i]) {
                visited[i] = true;
                queue[rear++] = i;
            }
        }
    }
}
```

## 3. 迷宫最短路径

在二维网格中找从起点到终点的最短路径。

```c
#include <stdio.h>
#include <stdbool.h>

#define MAXN 105

int maze[MAXN][MAXN];     // 0可通行, 1是墙
int dist[MAXN][MAXN];     // 到起点的距离
int n, m;                 // 行列
int dx[] = {0, 0, 1, -1};
int dy[] = {1, -1, 0, 0};

typedef struct { int x, y; } Point;
Point queue[MAXN * MAXN];

int bfs(int sx, int sy, int ex, int ey) {
    int front = 0, rear = 0;
    queue[rear++] = (Point){sx, sy};
    dist[sx][sy] = 0;

    while (front < rear) {
        Point cur = queue[front++];
        if (cur.x == ex && cur.y == ey) return dist[ex][ey];

        for (int d = 0; d < 4; d++) {
            int nx = cur.x + dx[d], ny = cur.y + dy[d];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m
                && !maze[nx][ny] && dist[nx][ny] == -1) {
                dist[nx][ny] = dist[cur.x][cur.y] + 1;
                queue[rear++] = (Point){nx, ny};
            }
        }
    }
    return -1;  // 无法到达
}
```

## 4. 多源BFS

从多个起点同时开始BFS，求每个位置到最近起点的距离。

```c
#include <stdio.h>
#include <string.h>

#define MAXN 1005

int dist[MAXN][MAXN];
int n, m;
typedef struct { int x, y; } Point;
Point queue[MAXN * MAXN];
int dx[] = {0, 0, 1, -1}, dy[] = {1, -1, 0, 0};

void multiSourceBFS(Point sources[], int k) {
    int front = 0, rear = 0;
    memset(dist, -1, sizeof(dist));
    for (int i = 0; i < k; i++) {
        queue[rear++] = sources[i];
        dist[sources[i].x][sources[i].y] = 0;
    }

    while (front < rear) {
        Point cur = queue[front++];
        for (int d = 0; d < 4; d++) {
            int nx = cur.x + dx[d], ny = cur.y + dy[d];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m
                && dist[nx][ny] == -1) {
                dist[nx][ny] = dist[cur.x][cur.y] + 1;
                queue[rear++] = (Point){nx, ny};
            }
        }
    }
}
```

## 5. BFS求最少操作次数

### 题目：水壶问题

有两个容量分别为x升和y升的水壶，通过倒水操作得到恰好z升水。

```c
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#define MAXN 10000

typedef struct { int a, b; } State;

bool visited[1005][1005];
State queue[MAXN];

bool canMeasure(int x, int y, int z) {
    if (z > x + y) return false;
    if (z == 0) return true;
    int front = 0, rear = 0;
    queue[rear++] = (State){0, 0};
    visited[0][0] = true;

    while (front < rear) {
        State cur = queue[front++];
        int a = cur.a, b = cur.b;
        if (a == z || b == z || a + b == z) return true;

        State nexts[] = {
            {x, b},         // 装满A
            {a, y},         // 装满B
            {0, b},         // 倒空A
            {a, 0},         // 倒空B
            {a - (y - b < a ? y - b : a), b + (y - b < a ? y - b : a)}, // A倒B
            {a + (x - a < b ? x - a : b), b - (x - a < b ? x - a : b)}, // B倒A
        };
        // 简化写法
        // A->B: pour = min(a, y-b)
        int pour = a < y - b ? a : y - b;
        nexts[4] = (State){a - pour, b + pour};
        pour = b < x - a ? b : x - a;
        nexts[5] = (State){a + pour, b - pour};

        for (int i = 0; i < 6; i++) {
            int na = nexts[i].a, nb = nexts[i].b;
            if (na >= 0 && na <= x && nb >= 0 && nb <= y && !visited[na][nb]) {
                visited[na][nb] = true;
                queue[rear++] = nexts[i];
            }
        }
    }
    return false;
}
```

## 6. 状态空间搜索

用BFS搜索状态空间，每个状态看作图中的一个节点。

```c
// 八数码问题: 3x3网格，通过滑动空格将初始状态变为目标状态
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    int board[3][3];  // 空格用0表示
    int x, y;         // 空格位置
    int step;
} State;

// 将状态编码为整数（康托展开）用于判重
int factorial[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};

int cantor(int board[3][3]) {
    int seq[9], k = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            seq[k++] = board[i][j];
    int ans = 0;
    for (int i = 0; i < 9; i++) {
        int cnt = 0;
        for (int j = i + 1; j < 9; j++)
            if (seq[j] < seq[i]) cnt++;
        ans += cnt * factorial[8 - i];
    }
    return ans;
}

int dx[] = {0, 0, 1, -1}, dy[] = {1, -1, 0, 0};

int bfsEightPuzzle(State start) {
    bool visited[362880];  // 9! = 362880
    memset(visited, false, sizeof(visited));
    // 使用数组模拟队列（实际应用中需大数组）
    // 此处省略队列实现，仅展示BFS框架
    // ...
    return -1;  // 无解
}
```

## 7. BFS vs DFS对比

| 特征 | BFS | DFS |
|------|-----|-----|
| 数据结构 | 队列 | 栈/递归 |
| 搜索顺序 | 逐层扩展 | 深入到底再回溯 |
| 最短路径 | 无权图保证最短 | 不保证 |
| 空间复杂度 | O(V) | O(V)，但实际更小 |
| 适用场景 | 最短路径、层级遍历 | 连通性、拓扑排序 |
