# A星寻路算法（A* Pathfinding）

## 1. 核心理论

A*（A-Star）算法是一种启发式图搜索算法，由Peter Hart、Nils Nilsson和Bertram Raphael于1968年在斯坦福研究院提出。它通过综合实际代价和启发式估计来指导搜索方向，在保证最优解的前提下大幅减少搜索空间。

### 1.1 评估函数

A*的核心是评估函数：

```
f(n) = g(n) + h(n)
```

- **g(n)**：从起点到节点n的**实际代价**（已走过的距离）
- **h(n)**：从节点n到目标的**启发式估计代价**（尚未走的距离估计）
- **f(n)**：经过节点n到达目标的**总估计代价**

A*始终从OpenList中选择f值最小的节点进行扩展，这保证了：
- 当h(n)可采纳（admissible）时，A*保证找到最优解
- 当h(n)一致（consistent）时，A*不会重复扩展已关闭的节点

### 1.2 完备性与最优性证明

**完备性**：如果存在解，A*一定能找到。因为A*本质上是广度优先搜索的推广，只要图有限且边权为正，就保证有限步内找到解。

**最优性**：假设h(n)是可采纳的（即h(n) <= h*(n)，其中h*(n)是节点n到目标的真实最短代价），则A*找到的第一个解一定是最优的。

**证明思路**：设最优路径为P*，代价为C*。假设A*先找到了一个次优解路径P，代价为C > C*。在P被找到时，最优路径上至少存在一个节点n尚未被扩展，且f(n) = g(n) + h(n) <= g(n) + h*(n) = C* < C。由于A*选择f最小的节点扩展，n的f值小于已找到解的f值(C)，A*必然先扩展n而非返回次优解，产生矛盾。

### 1.3 与Dijkstra和贪心搜索的关系

A*可以看作Dijkstra和贪心搜索的统一框架：

| 算法 | 评估函数 | 特性 |
|------|----------|------|
| Dijkstra | f(n) = g(n) | 保证最优，但搜索范围大 |
| 贪心搜索 | f(n) = h(n) | 效率高，但不保证最优 |
| A* | f(n) = g(n) + h(n) | 兼顾效率与最优性 |

当h(n)=0时，A*退化为Dijkstra；当g(n)=0时，A*退化为贪心搜索。

## 2. 启发函数设计

### 2.1 常用启发函数

```cpp
// 曼哈顿距离：适用于四方向网格移动
float Manhattan(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

// 欧几里得距离：适用于任意方向移动
float Euclidean(float x1, float y1, float x2, float y2) {
    return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// 切比雪夫距离：适用于八方向网格移动（对角代价=直线代价）
float Chebyshev(int x1, int y1, int x2, int y2) {
    return std::max(abs(x1 - x2), abs(y1 - y2));
}

// 对角距离：适用于八方向网格移动（对角代价=sqrt(2)）
float Octile(int x1, int y1, int x2, int y2) {
    int dx = abs(x1 - x2);
    int dy = abs(y1 - y2);
    float D = 1.0f;      // 直线代价
    float D2 = 1.414f;   // 对角代价
    return D * (dx + dy) + (D2 - 2.0f * D) * std::min(dx, dy);
}
```

### 2.2 可采纳性与一致性

- **可采纳（Admissible）**：h(n) <= h*(n) 对所有n成立，即从不高估到目标的代价
- **一致（Consistent / Monotone）**：对任意边(n, n')，h(n) <= cost(n, n') + h(n')

一致性蕴涵可采纳性。使用一致启发函数时，A*每个节点最多从OpenList取出一次，效率更高。

### 2.3 加权A*（Weighted A*）

通过权重w调整启发函数的影响：

```
f(n) = g(n) + w * h(n)
```

- w > 1：更贪婪，搜索更快但可能牺牲最优性（称为ε-admissible）
- w = 1：标准A*
- w = 0：退化为Dijkstra

实际游戏中常用w=1.5~2.5的加权A*换取速度，路径质量损失通常可接受。

## 3. 完整算法实现

### 3.1 数据结构定义

```cpp
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <functional>

struct GridPos {
    int x, y;
    bool operator==(const GridPos& o) const { return x == o.x && y == o.y; }
    bool operator!=(const GridPos& o) const { return !(*this == o); }
};

// 哈希函数
struct GridPosHash {
    size_t operator()(const GridPos& p) const {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 16);
    }
};

struct AStarNode {
    GridPos pos;
    float gCost;       // 从起点的实际代价
    float hCost;       // 到目标的启发估计
    float fCost;       // g + h
    AStarNode* parent;

    AStarNode(GridPos p, float g, float h, AStarNode* par = nullptr)
        : pos(p), gCost(g), hCost(h), fCost(g + h), parent(par) {}
};

// 最小堆比较器
struct NodeCompare {
    bool operator()(const AStarNode* a, const AStarNode* b) const {
        if (std::abs(a->fCost - b->fCost) < 0.001f)
            return a->hCost > b->hCost; // f相同时优先扩展h小的节点
        return a->fCost > b->fCost;
    }
};
```

### 3.2 网格地图

```cpp
class GridMap {
public:
    int width, height;
    std::vector<std::vector<bool>> walkable; // true=可通行

    GridMap(int w, int h) : width(w), height(h) {
        walkable.resize(h, std::vector<bool>(w, true));
    }

    bool IsWalkable(int x, int y) const {
        return x >= 0 && x < width && y >= 0 && y < height && walkable[y][x];
    }

    // 获取八方向邻居
    std::vector<GridPos> GetNeighbors(GridPos pos) const {
        static const int dx[] = {0, 1, 1, 1, 0, -1, -1, -1};
        static const int dy[] = {-1, -1, 0, 1, 1, 1, 0, -1};
        // 八方向的移动代价
        static const float costs[] = {1.0f, 1.414f, 1.0f, 1.414f,
                                       1.0f, 1.414f, 1.0f, 1.414f};

        std::vector<GridPos> result;
        for (int i = 0; i < 8; i++) {
            int nx = pos.x + dx[i];
            int ny = pos.y + dy[i];
            if (IsWalkable(nx, ny)) {
                // 对角移动需要检查两侧是否可通行（防止穿墙角）
                if (i % 2 == 1) { // 对角方向
                    if (!IsWalkable(pos.x + dx[i], pos.y) ||
                        !IsWalkable(pos.x, pos.y + dy[i]))
                        continue;
                }
                result.push_back({nx, ny});
            }
        }
        return result;
    }

    float GetMoveCost(GridPos from, GridPos to) const {
        return (from.x != to.x && from.y != to.y) ? 1.414f : 1.0f;
    }
};
```

### 3.3 核心搜索算法

```cpp
class AStarPathfinder {
public:
    std::vector<GridPos> FindPath(const GridMap& map, GridPos start, GridPos goal) {
        if (!map.IsWalkable(start.x, start.y) || !map.IsWalkable(goal.x, goal.y))
            return {};

        // OpenList: 最小堆
        std::priority_queue<AStarNode*, std::vector<AStarNode*>, NodeCompare> openList;
        // 用于快速查找OpenList中是否存在某节点
        std::unordered_map<GridPos, AStarNode*, GridPosHash> openMap;
        // CloseList: 已扩展的节点
        std::unordered_set<GridPos, GridPosHash> closeList;

        float h = Octile(start.x, start.y, goal.x, goal.y);
        AStarNode* startNode = new AStarNode(start, 0.0f, h);
        openList.push(startNode);
        openMap[start] = startNode;

        while (!openList.empty()) {
            AStarNode* current = openList.top();
            openList.pop();

            // 跳过已关闭的节点（惰性删除）
            if (closeList.count(current->pos)) {
                delete current;
                continue;
            }

            // 找到目标
            if (current->pos == goal) {
                std::vector<GridPos> path = ReconstructPath(current);
                Cleanup(openList, openMap);
                return path;
            }

            closeList.insert(current->pos);

            // 扩展邻居
            for (const GridPos& neighborPos : map.GetNeighbors(current->pos)) {
                if (closeList.count(neighborPos)) continue;

                float tentativeG = current->gCost + map.GetMoveCost(current->pos, neighborPos);

                auto it = openMap.find(neighborPos);
                if (it != openMap.end()) {
                    // 节点已在OpenList中，检查是否需要更新
                    if (tentativeG < it->second->gCost) {
                        it->second->gCost = tentativeG;
                        it->second->fCost = tentativeG + it->second->hCost;
                        it->second->parent = current;
                        // 重新插入堆（惰性更新策略）
                        AStarNode* updated = new AStarNode(
                            neighborPos, tentativeG, it->second->hCost, current);
                        openList.push(updated);
                        openMap[neighborPos] = updated;
                    }
                } else {
                    // 新节点
                    float h = Octile(neighborPos.x, neighborPos.y, goal.x, goal.y);
                    AStarNode* newNode = new AStarNode(neighborPos, tentativeG, h, current);
                    openList.push(newNode);
                    openMap[neighborPos] = newNode;
                }
            }
        }

        Cleanup(openList, openMap);
        return {}; // 无路径
    }

private:
    std::vector<GridPos> ReconstructPath(AStarNode* node) {
        std::vector<GridPos> path;
        while (node) {
            path.push_back(node->pos);
            node = node->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    void Cleanup(std::priority_queue<AStarNode*, std::vector<AStarNode*>, NodeCompare>& q,
                 std::unordered_map<GridPos, AStarNode*, GridPosHash>& m) {
        while (!q.empty()) { delete q.top(); q.pop(); }
        m.clear();
    }
};
```

## 4. 高级变体算法

### 4.1 Jump Point Search（JPS）

JPS是Harabor和Grastien在2011年提出的A*优化算法，专门针对**均匀代价网格**。核心思想是通过"跳点"剪枝，跳过对称路径中的中间节点，将搜索节点数减少一个数量级。

**跳点定义**：一个节点x是跳点，当且仅当满足以下条件之一：
1. x是起点或目标
2. x至少有一个强迫邻居（forced neighbor）
3. x在水平/垂直方向的递归搜索路径上

```cpp
class JPSPathfinder {
public:
    std::vector<GridPos> FindPath(const GridMap& map, GridPos start, GridPos goal) {
        std::priority_queue<JPSNode*, std::vector<JPSNode*>, JPSCompare> openList;
        std::unordered_map<GridPos, float, GridPosHash> gScores;

        gScores[start] = 0;
        openList.push(new JPSNode{start, 0, Octile(start.x, start.y, goal.x, goal.y), nullptr});

        while (!openList.empty()) {
            JPSNode* current = openList.top();
            openList.pop();

            if (current->pos == goal) {
                // 回溯路径...
                return ReconstructPath(current);
            }

            // 识别后继跳点
            auto successors = IdentifySuccessors(map, current->pos, goal);
            for (auto& [sucPos, dir] : successors) {
                float newG = current->g + Distance(current->pos, sucPos);
                auto it = gScores.find(sucPos);
                if (it == gScores.end() || newG < it->second) {
                    gScores[sucPos] = newG;
                    float h = Octile(sucPos.x, sucPos.y, goal.x, goal.y);
                    openList.push(new JPSNode{sucPos, newG, h, current});
                }
            }
        }
        return {};
    }

private:
    struct JPSNode {
        GridPos pos;
        float g, f;
        JPSNode* parent;
    };

    struct JPSCompare {
        bool operator()(const JPSNode* a, const JPSNode* b) const {
            return a->f > b->f;
        }
    };

    // 跳跃函数：沿指定方向递归寻找跳点
    GridPos Jump(const GridMap& map, GridPos pos, int dx, int dy, GridPos goal) {
        int nx = pos.x + dx, ny = pos.y + dy;
        if (!map.IsWalkable(nx, ny)) return {-1, -1}; // 无效

        GridPos next = {nx, ny};
        if (next == goal) return next;

        // 检查强迫邻居
        if (HasForcedNeighbors(map, next, dx, dy)) return next;

        // 对角移动：先沿对角跳，再沿两个分量跳
        if (dx != 0 && dy != 0) {
            if (Jump(map, next, dx, 0, goal).x != -1) return next;
            if (Jump(map, next, 0, dy, goal).x != -1) return next;
        }

        return Jump(map, next, dx, dy, goal); // 继续递归
    }

    bool HasForcedNeighbors(const GridMap& map, GridPos pos, int dx, int dy) {
        // 检查是否存在强迫邻居
        if (dx != 0 && dy != 0) { // 对角移动
            return (!map.IsWalkable(pos.x - dx, pos.y) && map.IsWalkable(pos.x - dx, pos.y + dy)) ||
                   (!map.IsWalkable(pos.x, pos.y - dy) && map.IsWalkable(pos.x + dx, pos.y - dy));
        } else if (dx != 0) { // 水平移动
            return (!map.IsWalkable(pos.x, pos.y + 1) && map.IsWalkable(pos.x + dx, pos.y + 1)) ||
                   (!map.IsWalkable(pos.x, pos.y - 1) && map.IsWalkable(pos.x + dx, pos.y - 1));
        } else { // 垂直移动
            return (!map.IsWalkable(pos.x + 1, pos.y) && map.IsWalkable(pos.x + 1, pos.y + dy)) ||
                   (!map.IsWalkable(pos.x - 1, pos.y) && map.IsWalkable(pos.x - 1, pos.y + dy));
        }
    }

    std::vector<GridPos> ReconstructPath(JPSNode* node) {
        std::vector<GridPos> path;
        while (node) { path.push_back(node->pos); node = node->parent; }
        std::reverse(path.begin(), path.end());
        return path;
    }
};
```

**JPS性能对比**：在1000x1000的开放网格地图上，JPS比标准A*减少约80-95%的扩展节点数，速度提升5-20倍。

### 4.2 分层A*（Hierarchical A*）

适用于超大地图（如开放世界游戏），将地图划分为多个层级：

```
高层: 世界区域图（区域间连通性）
中层: 地块图（tile间路径）
低层: 精确网格（局部避障）
```

**算法流程**：
1. 在高层区域图上用A*找到区域路径
2. 在中层地块图上细化区域间的连接点
3. 在低层网格上精确计算路径

**优势**：搜索复杂度从O(V log V)降低到O(k * (V/k) log(V/k))，其中k为层数。

### 4.3 流场（Flow Field）

流场适用于"多源到单目标"的寻路场景（如RTS中大量单位前往同一目标）：

```cpp
class FlowField {
public:
    int width, height;
    std::vector<Vec2> directions; // 每个格子的移动方向

    void Generate(const GridMap& map, GridPos goal) {
        // 从目标反向BFS，计算距离场
        std::vector<float> distances(width * height, FLT_MAX);
        std::queue<GridPos> queue;
        distances[goal.y * width + goal.x] = 0;
        queue.push(goal);

        while (!queue.empty()) {
            GridPos cur = queue.front(); queue.pop();
            float curDist = distances[cur.y * width + cur.x];

            for (auto& nb : map.GetNeighbors(cur)) {
                float newDist = curDist + map.GetMoveCost(cur, nb);
                int idx = nb.y * width + nb.x;
                if (newDist < distances[idx]) {
                    distances[idx] = newDist;
                    queue.push(nb);
                }
            }
        }

        // 根据距离场生成方向场
        directions.resize(width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float minDist = distances[y * width + x];
                GridPos best = {x, y};

                for (auto& nb : map.GetNeighbors({x, y})) {
                    int idx = nb.y * width + nb.x;
                    if (distances[idx] < minDist) {
                        minDist = distances[idx];
                        best = nb;
                    }
                }

                Vec2 dir = Vec2{(float)(best.x - x), (float)(best.y - y)};
                float len = sqrtf(dir.x * dir.x + dir.y * dir.y);
                if (len > 0) { dir.x /= len; dir.y /= len; }
                directions[y * width + x] = dir;
            }
        }
    }

    // 查询：任意位置的移动方向，O(1)
    Vec2 GetDirection(GridPos pos) const {
        return directions[pos.y * width + pos.x];
    }
};
```

**流场 vs A*对比**：

| 维度 | A* | 流场 |
|------|-----|------|
| 单位寻路 | 每个单位独立搜索 | 预计算一次，所有单位共用 |
| 时间复杂度 | O(V log V) 每查询 | O(V log V) 预计算 + O(1) 查询 |
| 适用场景 | 不同目标点 | 大量单位同一目标 |
| 动态障碍 | 重新搜索 | 重新生成流场 |
| 内存 | 低 | O(V) 方向存储 |

## 5. 路径平滑技术

原始A*输出的路径沿网格拐角移动，不够自然。常用平滑方法：

### 5.1 视线检测平滑（Line-of-Sight Smoothing）

```cpp
std::vector<Vec3> SmoothPath(const std::vector<Vec3>& rawPath,
                              std::function<bool(Vec3, Vec3)> hasLineOfSight) {
    if (rawPath.size() <= 2) return rawPath;

    std::vector<Vec3> smoothed;
    smoothed.push_back(rawPath[0]);

    int current = 0;
    while (current < (int)rawPath.size() - 1) {
        int farthest = current + 1;
        for (int i = (int)rawPath.size() - 1; i > current + 1; i--) {
            if (hasLineOfSight(rawPath[current], rawPath[i])) {
                farthest = i;
                break;
            }
        }
        smoothed.push_back(rawPath[farthest]);
        current = farthest;
    }
    return smoothed;
}
```

### 5.2 Catmull-Rom样条平滑

```cpp
Vec3 CatmullRom(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5f * (
        (2.0f * p1) +
        (-p0 + p2) * t +
        (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
        (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
    );
}

std::vector<Vec3> SmoothWithSpline(const std::vector<Vec3>& path, int subdivisions = 10) {
    std::vector<Vec3> result;
    for (int i = 0; i < (int)path.size() - 1; i++) {
        Vec3 p0 = path[std::max(0, i - 1)];
        Vec3 p1 = path[i];
        Vec3 p2 = path[std::min((int)path.size() - 1, i + 1)];
        Vec3 p3 = path[std::min((int)path.size() - 1, i + 2)];

        for (int j = 0; j < subdivisions; j++) {
            float t = (float)j / subdivisions;
            result.push_back(CatmullRom(p0, p1, p2, p3, t));
        }
    }
    result.push_back(path.back());
    return result;
}
```

## 6. 性能分析

### 6.1 时间复杂度

| 算法 | 最好情况 | 平均情况 | 最差情况 |
|------|----------|----------|----------|
| A* | O(b^d) | O(b^(w*d)) | O(b^m) |
| JPS | O(b^(d/2)) | O(b^(w*d/3)) | O(b^m) |
| Dijkstra | O(V + E log V) | O(V + E log V) | O(V + E log V) |
| 流场 | - | O(V + E) | O(V + E) |

其中b为分支因子，d为最优解深度，w为启发函数权重，m为最大搜索深度，V为节点数，E为边数。

### 6.2 空间复杂度

所有变体的空间复杂度均为O(V)，需要存储OpenList和CloseList中的所有节点。

### 6.3 OpenList数据结构选择

| 数据结构 | 取最小 | 插入 | 更新 | 适用场景 |
|----------|--------|------|------|----------|
| 二叉堆 | O(log n) | O(log n) | O(n) | 通用场景 |
| 斐波那契堆 | O(1)摊还 | O(1) | O(1) | 大规模图 |
| 有序数组 | O(1) | O(n) | O(n) | 小规模图 |
| Bucket Queue | O(1) | O(1) | O(1) | 整数权值 |

## 7. 游戏引擎实现

### 7.1 Unity实现

```csharp
// Unity中的A*实现
using UnityEngine;
using System.Collections.Generic;

public class AStarUnity : MonoBehaviour {
    public Grid grid; // Unity Tilemap Grid
    public Transform seeker, target;

    private List<Vector3Int> openSet = new List<Vector3Int>();
    private HashSet<Vector3Int> closedSet = new HashSet<Vector3Int>();
    private Dictionary<Vector3Int, Vector3Int> cameFrom = new Dictionary<Vector3Int, Vector3Int>();
    private Dictionary<Vector3Int, float> gScore = new Dictionary<Vector3Int, float>();
    private Dictionary<Vector3Int, float> fScore = new Dictionary<Vector3Int, float>();

    void Update() {
        if (Input.GetKeyDown(KeyCode.Space)) {
            Vector3Int start = grid.WorldToCell(seeker.position);
            Vector3Int goal = grid.WorldToCell(target.position);
            List<Vector3Int> path = FindPath(start, goal);
            // 使用LineRenderer绘制路径
            LineRenderer lr = GetComponent<LineRenderer>();
            lr.positionCount = path.Count;
            for (int i = 0; i < path.Count; i++) {
                lr.SetPosition(i, grid.CellToWorld(path[i]) + new Vector3(0.5f, 0.5f, 0));
            }
        }
    }

    List<Vector3Int> FindPath(Vector3Int start, Vector3Int goal) {
        openSet.Clear(); closedSet.Clear();
        cameFrom.Clear(); gScore.Clear(); fScore.Clear();

        gScore[start] = 0;
        fScore[start] = Heuristic(start, goal);
        openSet.Add(start);

        while (openSet.Count > 0) {
            Vector3Int current = GetLowestF();
            if (current == goal) return ReconstructPath(cameFrom, current);

            openSet.Remove(current);
            closedSet.Add(current);

            foreach (var neighbor in GetNeighbors(current)) {
                if (closedSet.Contains(neighbor)) continue;
                float tentativeG = gScore[current] + 1;

                if (!gScore.ContainsKey(neighbor) || tentativeG < gScore[neighbor]) {
                    cameFrom[neighbor] = current;
                    gScore[neighbor] = tentativeG;
                    fScore[neighbor] = tentativeG + Heuristic(neighbor, goal);
                    if (!openSet.Contains(neighbor)) openSet.Add(neighbor);
                }
            }
        }
        return new List<Vector3Int>();
    }

    float Heuristic(Vector3Int a, Vector3Int b) {
        return Mathf.Abs(a.x - b.x) + Mathf.Abs(a.y - b.y); // 曼哈顿
    }

    Vector3Int GetLowestF() {
        Vector3Int best = openSet[0];
        float bestF = fScore.ContainsKey(best) ? fScore[best] : float.MaxValue;
        foreach (var n in openSet) {
            float f = fScore.ContainsKey(n) ? fScore[n] : float.MaxValue;
            if (f < bestF) { bestF = f; best = n; }
        }
        return best;
    }

    List<Vector3Int> GetNeighbors(Vector3Int pos) {
        // 4方向邻居
        Vector3Int[] dirs = { Vector3Int.up, Vector3Int.down,
                              Vector3Int.left, Vector3Int.right };
        List<Vector3Int> result = new List<Vector3Int>();
        foreach (var d in dirs) {
            Vector3Int n = pos + d;
            // 检查Tilemap是否可通行
            if (grid.GetComponent<UnityEngine.Tilemaps.Tilemap>().HasTile(n))
                result.Add(n);
        }
        return result;
    }

    List<Vector3Int> ReconstructPath(Dictionary<Vector3Int, Vector3Int> from, Vector3Int cur) {
        List<Vector3Int> path = new List<Vector3Int> { cur };
        while (from.ContainsKey(cur)) { cur = from[cur]; path.Add(cur); }
        path.Reverse();
        return path;
    }
}
```

### 7.2 Unreal Engine实现

UE内置了`UNavigationSystemV1`用于寻路，但了解底层实现有助于自定义：

```cpp
// UE5 A*自定义实现示例
#include "NavigationPath.h"
#include "NavFilters/NavigationQueryFilter.h"

UCLASS()
class UMyAStarPathfinding : public UObject {
    GENERATED_BODY()

public:
    TArray<FVector> FindPath(UWorld* World, FVector Start, FVector End) {
        // 使用UE内置导航系统
        UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(World);
        if (!NavSys) return {};

        FNavLocation StartNavLoc, EndNavLoc;
        NavSys->ProjectPointToNavigation(Start, StartNavLoc);
        NavSys->ProjectPointToNavigation(End, EndNavLoc);

        // 也可直接使用FindPathToLocationSynchronously
        UNavigationPath* NavPath = NavSys->FindPathToLocationSynchronously(
            World, StartNavLoc.Location, EndNavLoc.Location);

        if (NavPath && NavPath->IsValid()) {
            return NavPath->PathPoints;
        }
        return {};
    }

    // 自定义A*（用于特殊需求）
    TArray<FIntPoint> CustomAStar(
        const TArray<TArray<bool>>& Grid, FIntPoint Start, FIntPoint Goal)
    {
        int32 W = Grid.Num(), H = Grid[0].Num();
        auto Heuristic = [](FIntPoint A, FIntPoint B) -> float {
            return FMath::Abs(A.X - B.X) + FMath::Abs(A.Y - B.Y);
        };

        TMap<FIntPoint, float> GScore;
        TMap<FIntPoint, FIntPoint> CameFrom;

        auto Cmp = [](const TPair<float, FIntPoint>& A, const TPair<float, FIntPoint>& B) {
            return A.Key > B.Key;
        };
        TArray<TPair<float, FIntPoint>> OpenList;

        GScore.Add(Start, 0);
        OpenList.Emplace(Heuristic(Start, Goal), Start);

        while (OpenList.Num() > 0) {
            OpenList.Sort(Cmp);
            auto [F, Current] = OpenList.Pop();

            if (Current == Goal) {
                // 回溯路径
                TArray<FIntPoint> Path;
                FIntPoint Cur = Goal;
                while (Cur != Start) { Path.Add(Cur); Cur = CameFrom[Cur]; }
                Path.Add(Start);
                Algo::Reverse(Path);
                return Path;
            }

            static FIntPoint Dirs[] = {{0,1},{1,0},{0,-1},{-1,0}};
            for (auto& D : Dirs) {
                FIntPoint Nb = Current + D;
                if (Nb.X < 0 || Nb.X >= W || Nb.Y < 0 || Nb.Y >= H) continue;
                if (!Grid[Nb.X][Nb.Y]) continue;

                float NewG = GScore[Current] + 1;
                if (!GScore.Contains(Nb) || NewG < GScore[Nb]) {
                    GScore.Add(Nb, NewG);
                    CameFrom.Add(Nb, Current);
                    OpenList.Emplace(NewG + Heuristic(Nb, Goal), Nb);
                }
            }
        }
        return {};
    }
};
```

## 8. 调试与可视化

```cpp
class AStarDebugger {
public:
    // 绘制搜索过程
    static void DrawSearchVisualization(
        const std::unordered_set<GridPos, GridPosHash>& openSet,
        const std::unordered_set<GridPos, GridPosHash>& closedSet,
        const std::vector<GridPos>& finalPath,
        float cellSize)
    {
        // OpenList节点用黄色绘制
        for (const auto& pos : openSet) {
            DrawSquare(pos.x * cellSize, pos.y * cellSize, cellSize, Color::Yellow, 0.3f);
        }
        // CloseList节点用红色绘制
        for (const auto& pos : closedSet) {
            DrawSquare(pos.x * cellSize, pos.y * cellSize, cellSize, Color::Red, 0.3f);
        }
        // 最终路径用绿色高亮
        for (size_t i = 0; i < finalPath.size() - 1; i++) {
            Vec3 a = {(float)finalPath[i].x * cellSize, 0, (float)finalPath[i].y * cellSize};
            Vec3 b = {(float)finalPath[i+1].x * cellSize, 0, (float)finalPath[i+1].y * cellSize};
            DrawLine(a, b, Color::Green, 3.0f);
        }
    }

    // 绘制f/g/h值热力图
    static void DrawHeatmap(
        const std::unordered_map<GridPos, AStarNode*, GridPosHash>& nodes,
        GridPos goal, float cellSize)
    {
        float maxF = 0;
        for (const auto& [pos, node] : nodes) {
            maxF = std::max(maxF, node->fCost);
        }
        for (const auto& [pos, node] : nodes) {
            float t = node->fCost / maxF; // 归一化到[0,1]
            Color c = ColorLerp(Color::Blue, Color::Red, t);
            DrawSquare(pos.x * cellSize, pos.y * cellSize, cellSize, c, 0.5f);
        }
    }
};
```

## 9. 实际游戏案例分析

### 案例1：星际争霸2的寻路系统

星际争霸2处理同时数百个单位的实时寻路，采用分层方案：
- **宏观路径**：用改进的HPA*在区域图上规划大致路线
- **微观避障**：局部使用RVO（互易速度障碍）处理单位间碰撞
- **路径缓存**：相同起终点的查询结果缓存，减少重复计算
- **异步更新**：寻路请求分布到多帧执行，避免单帧卡顿

### 案例2：暗黑破坏神系列

暗黑破坏神作为ARPG，玩家和怪物需要实时寻路：
- 使用八方向A*配合对角距离启发函数
- 动态障碍（如玩家放置的障碍物）通过动态更新网格处理
- 路径平滑使用Catmull-Rom样条，使角色移动更流畅
- 怪物密度高时使用流场替代A*优化性能

### 案例3：文明6的战略寻路

文明6是回合制策略游戏，单位需要跨越数百格的大陆：
- 使用带地形代价权重的A*
- 不同单位类型有不同可通行地形（海军、陆军、飞行单位）
- 寻路计算在回合切换时异步执行
- 长距离路径使用分层A*，先在大陆级别规划，再细化局部路径
