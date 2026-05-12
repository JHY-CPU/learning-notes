# 群体AI与蜂群行为

## 1. 核心理论

### 1.1 什么是群体AI

群体AI（Swarm Intelligence / Crowd Simulation）模拟大规模实体的协调运动行为。核心思想是：每个个体仅依赖**局部邻居信息**做出运动决策，无需中央协调，却能涌现出有序、自然的群体行为。这种自组织现象在自然界中广泛存在：鸟群迁徙、鱼群游动、蚁群觅食。

Craig Reynolds在1986年的论文"Flocks, Herds, and Schools: A Distributed Behavioral Model"中提出了经典的**Boids算法**，成为群体模拟的基石。

### 1.2 涌现行为

群体AI的核心特征是**涌现**（Emergence）：简单的局部规则产生复杂的全局行为。每只Boid只知道自己周围几只邻居的信息，但群体整体呈现出：

- 有序的队列运动
- 自然的避障流线
- 动态的群体形状变化
- 智能的分裂与合并

这种涌现行为的关键在于**没有个体知道全局计划**，秩序完全来自局部交互。

### 1.3 设计哲学

群体AI的设计遵循三个原则：

1. **去中心化**：无全局控制器，每个个体自主决策
2. **局部感知**：每个个体只感知有限范围内的邻居
3. **简单规则**：个体行为规则简单，复杂度从交互中涌现

## 2. Boids三规则

### 2.1 分离（Separation）

远离过近的邻居，防止碰撞和重叠。

```cpp
struct Boid {
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    float maxSpeed;
    float maxForce;
    float perceptionRadius;
};

Vec3 Separation(const Boid& boid, const std::vector<Boid*>& neighbors, float separationRadius) {
    Vec3 steer = {0, 0, 0};
    int count = 0;

    for (const Boid* other : neighbors) {
        float dist = Distance(boid.position, other->position);
        if (dist > 0 && dist < separationRadius) {
            // 方向远离邻居，按距离加权（越近排斥力越大）
            Vec3 diff = (boid.position - other->position).Normalized();
            diff = diff * (1.0f / dist); // 距离加权
            steer = steer + diff;
            count++;
        }
    }

    if (count > 0) {
        steer = steer * (1.0f / count);  // 平均
        steer = steer.Normalized() * boid.maxSpeed;
        steer = steer - boid.velocity;    // 转向力
        steer = LimitMagnitude(steer, boid.maxForce);
    }
    return steer;
}
```

### 2.2 对齐（Alignment）

朝邻居的平均方向调整，保持队形一致。

```cpp
Vec3 Alignment(const Boid& boid, const std::vector<Boid*>& neighbors, float alignRadius) {
    Vec3 avgVelocity = {0, 0, 0};
    int count = 0;

    for (const Boid* other : neighbors) {
        float dist = Distance(boid.position, other->position);
        if (dist > 0 && dist < alignRadius) {
            avgVelocity = avgVelocity + other->velocity;
            count++;
        }
    }

    if (count > 0) {
        avgVelocity = avgVelocity * (1.0f / count);
        avgVelocity = avgVelocity.Normalized() * boid.maxSpeed;
        Vec3 steer = avgVelocity - boid.velocity;
        return LimitMagnitude(steer, boid.maxForce);
    }
    return {0, 0, 0};
}
```

### 2.3 聚集（Cohesion）

向邻居的质心移动，保持群体聚合。

```cpp
Vec3 Cohesion(const Boid& boid, const std::vector<Boid*>& neighbors, float cohesionRadius) {
    Vec3 center = {0, 0, 0};
    int count = 0;

    for (const Boid* other : neighbors) {
        float dist = Distance(boid.position, other->position);
        if (dist > 0 && dist < cohesionRadius) {
            center = center + other->position;
            count++;
        }
    }

    if (count > 0) {
        center = center * (1.0f / count);
        // 朝质心方向移动（seek行为）
        Vec3 desired = (center - boid.position).Normalized() * boid.maxSpeed;
        Vec3 steer = desired - boid.velocity;
        return LimitMagnitude(steer, boid.maxForce);
    }
    return {0, 0, 0};
}
```

### 2.4 完整更新循环

```cpp
class BoidSimulation {
    std::vector<Boid> boids;

    void Update(float dt) {
        for (Boid& boid : boids) {
            // 获取邻居（使用空间分区优化）
            std::vector<Boid*> neighbors = GetNeighbors(boid, 50.0f);

            // 计算三种基本力
            Vec3 sep = Separation(boid, neighbors, 25.0f) * 1.5f;
            Vec3 ali = Alignment(boid, neighbors, 50.0f) * 1.0f;
            Vec3 coh = Cohesion(boid, neighbors, 50.0f) * 1.0f;

            // 可选附加力
            Vec3 avoid = AvoidObstacles(boid) * 2.0f;
            Vec3 boundary = BoundaryForce(boid) * 1.0f;

            // 合力
            boid.acceleration = sep + ali + coh + avoid + boundary;

            // 物理更新
            boid.velocity = boid.velocity + boid.acceleration * dt;
            boid.velocity = LimitMagnitude(boid.velocity, boid.maxSpeed);
            boid.position = boid.position + boid.velocity * dt;

            // 重置加速度
            boid.acceleration = {0, 0, 0};
        }
    }
};
```

## 3. 空间分区优化

### 3.1 问题

朴素的邻居搜索是O(N^2)复杂度——每只Boid需要与所有其他Boid比较距离。当Boid数量达到数千时，帧率严重下降。

### 3.2 空间哈希（Spatial Hashing）

将空间划分为固定大小的网格单元，每个Boid根据位置分配到对应的单元格：

```cpp
class SpatialHash {
    float cellSize;
    std::unordered_map<int64_t, std::vector<Boid*>> grid;

    int64_t HashKey(int x, int y, int z) const {
        // 空间哈希函数
        return ((int64_t)(x + 73856093) * 19349663 ^
                (int64_t)(y + 83492791) * 83492791 ^
                (int64_t)(z + 45678901) * 45678901);
    }

public:
    SpatialHash(float cell) : cellSize(cell) {}

    void Clear() { grid.clear(); }

    void Insert(Boid* boid) {
        int cx = (int)floorf(boid->position.x / cellSize);
        int cy = (int)floorf(boid->position.y / cellSize);
        int cz = (int)floorf(boid->position.z / cellSize);
        grid[HashKey(cx, cy, cz)].push_back(boid);
    }

    std::vector<Boid*> Query(const Vec3& pos, float radius) const {
        std::vector<Boid*> result;
        int cr = (int)ceilf(radius / cellSize);

        int cx = (int)floorf(pos.x / cellSize);
        int cy = (int)floorf(pos.y / cellSize);
        int cz = (int)floorf(pos.z / cellSize);

        // 搜索周围所有单元格
        for (int dx = -cr; dx <= cr; dx++) {
            for (int dy = -cr; dy <= cr; dy++) {
                for (int dz = -cr; dz <= cr; dz++) {
                    int64_t key = HashKey(cx+dx, cy+dy, cz+dz);
                    auto it = grid.find(key);
                    if (it != grid.end()) {
                        for (Boid* b : it->second) {
                            if (Distance(pos, b->position) <= radius) {
                                result.push_back(b);
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
};
```

### 3.3 性能对比

| Boid数量 | 暴力O(N^2) | 空间哈希O(N*k) | 提升倍数 |
|----------|-----------|---------------|---------|
| 100 | ~5K次比较 | ~500次比较 | 10x |
| 1,000 | ~500K | ~5K | 100x |
| 10,000 | ~50M | ~50K | 1000x |
| 100,000 | ~5B | ~500K | 10,000x |

k为平均邻居数，通常在5-30之间。

## 4. 障碍物避让

### 4.1 排斥力避让

最简单的方法：对附近障碍物施加排斥力。

```cpp
Vec3 AvoidObstacles(const Boid& boid, const std::vector<Obstacle>& obstacles) {
    Vec3 steer = {0, 0, 0};

    for (const Obstacle& obs : obstacles) {
        float dist = Distance(boid.position, obs.position);
        float avoidDist = obs.radius + boid.perceptionRadius * 0.5f;

        if (dist < avoidDist) {
            Vec3 diff = (boid.position - obs.position).Normalized();
            diff = diff * (1.0f / dist); // 距离越近排斥力越大
            steer = steer + diff;
        }
    }

    return LimitMagnitude(steer, boid.maxForce * 2.0f); // 避障力权重更大
}
```

### 4.2 预测性避让（Predictive Avoidance）

检测Boid前方是否有障碍物，提前调整方向：

```cpp
Vec3 PredictiveAvoidance(const Boid& boid, const std::vector<Obstacle>& obstacles,
                          float lookAheadTime = 2.0f) {
    // 预测未来位置
    Vec3 futurePos = boid.position + boid.velocity * lookAheadTime;

    for (const Obstacle& obs : obstacles) {
        // 检查预测路径是否与障碍物相交
        float distToFuture = Distance(futurePos, obs.position);
        if (distToFuture < obs.radius + 2.0f) {
            // 计算避让方向
            Vec3 avoidDir;
            Vec3 toObs = obs.position - boid.position;
            Vec3 right = Cross(boid.velocity, Vec3{0, 1, 0}).Normalized();

            // 向侧方避让
            if (Dot(toObs, right) > 0) avoidDir = right * -1.0f;
            else avoidDir = right;

            return avoidDir.Normalized() * boid.maxForce * 2.0f;
        }
    }
    return {0, 0, 0};
}
```

### 4.3 射线投射避让

```cpp
Vec3 RaycastAvoidance(const Boid& boid, float rayLength,
                       std::function<bool(Vec3, Vec3)> raycast) {
    // 向前方投射多条射线
    Vec3 forward = boid.velocity.Normalized();
    Vec3 right = Cross(forward, Vec3{0, 1, 0}).Normalized();
    Vec3 up = Cross(right, forward).Normalized();

    // 射线方向：前、左前、右前、上、下
    Vec3 rayDirs[] = {
        forward,
        (forward + right * 0.5f).Normalized(),
        (forward - right * 0.5f).Normalized(),
        (forward + up * 0.3f).Normalized(),
        (forward - up * 0.3f).Normalized()
    };

    Vec3 steer = {0, 0, 0};
    for (const Vec3& dir : rayDirs) {
        if (raycast(boid.position, dir)) {
            // 射线被阻挡，远离此方向
            steer = steer - dir * boid.maxForce;
        }
    }

    return LimitMagnitude(steer, boid.maxForce * 2.0f);
}
```

## 5. 高级群体行为

### 5.1 领导者跟随（Leader Following）

部分个体作为领导者沿预定路径移动，其他个体跟随领导者：

```cpp
class LeaderFollowing {
    Boid* leader;
    std::vector<Vec3> leaderPath;
    int pathIndex = 0;

public:
    Vec3 LeaderBehavior(const Boid& boid, float dt) {
        if (boid.position == leader->position) {
            // 当前Boid是领导者：沿路径移动
            Vec3 target = leaderPath[pathIndex];
            Vec3 desired = (target - boid.position).Normalized() * boid.maxSpeed;
            if (Distance(boid.position, target) < 2.0f) {
                pathIndex = (pathIndex + 1) % leaderPath.size();
            }
            return desired - boid.velocity;
        } else {
            // 非领导者：跟随 + 保持距离
            Vec3 follow = Seek(boid, leader->position);
            Vec3 separation = Separation(boid, GetNeighbors(boid, 15.0f), 10.0f);
            return follow * 1.0f + separation * 1.5f;
        }
    }
};
```

### 5.2 编队运动（Formation Movement）

群体按照指定队形移动（如V字形、圆形、网格）：

```cpp
struct FormationSlot {
    Vec3 localOffset; // 相对于编队中心的偏移
    int assignedBoid; // 分配的Boid索引
};

class FormationManager {
    Vec3 formationCenter;
    Vec3 formationDirection;
    std::vector<FormationSlot> slots;

public:
    void UpdateFormation(const Vec3& newCenter, const Vec3& newDir) {
        formationCenter = newCenter;
        formationDirection = newDir.Normalized();
    }

    // V字形编队
    void SetVFormation(int count, float spacing) {
        slots.clear();
        for (int i = 0; i < count; i++) {
            FormationSlot slot;
            int wing = (i % 2 == 0) ? 1 : -1;
            int depth = (i + 1) / 2;
            slot.localOffset = Vec3{
                -depth * spacing,              // 后退
                0,
                wing * depth * spacing * 0.5f  // 展开
            };
            slot.assignedBoid = i;
            slots.push_back(slot);
        }
    }

    Vec3 GetSlotWorldPosition(int slotIndex) const {
        if (slotIndex >= (int)slots.size()) return formationCenter;
        // 将局部偏移旋转到编队方向
        Vec3 offset = RotateToDirection(slots[slotIndex].localOffset, formationDirection);
        return formationCenter + offset;
    }

    // Boid的编队跟随力
    Vec3 FormationForce(const Boid& boid, int slotIndex) const {
        Vec3 target = GetSlotWorldPosition(slotIndex);
        float dist = Distance(boid.position, target);
        if (dist < 1.0f) return {0, 0, 0}; // 已到位
        return Seek(boid, target) * (dist / 10.0f); // 距离越远吸引力越大
    }
};
```

### 5.3 流场导航（Flow Field Navigation）

群体使用共享的流场导航，适合大量单位前往同一目标：

```cpp
class FlowField {
    int width, height;
    float cellSize;
    std::vector<Vec3> directions;
    std::vector<float> costs;

public:
    void Generate(const GridMap& grid, Vec3 goal) {
        // Dijkstra从目标反向计算代价场
        std::queue<Vec2i> queue;
        costs.assign(width * height, FLT_MAX);

        int gx = (int)(goal.x / cellSize);
        int gy = (int)(goal.z / cellSize);
        costs[gy * width + gx] = 0;
        queue.push({gx, gy});

        static int dx[] = {0, 1, 0, -1};
        static int dy[] = {-1, 0, 1, 0};

        while (!queue.empty()) {
            Vec2i cur = queue.front(); queue.pop();
            float curCost = costs[cur.y * width + cur.x];

            for (int i = 0; i < 4; i++) {
                int nx = cur.x + dx[i], ny = cur.y + dy[i];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                if (!grid.IsWalkable(nx * cellSize, ny * cellSize)) continue;

                float newCost = curCost + cellSize;
                int idx = ny * width + nx;
                if (newCost < costs[idx]) {
                    costs[idx] = newCost;
                    queue.push({nx, ny});
                }
            }
        }

        // 生成方向场：每个格子指向代价最低的邻居
        directions.resize(width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float minCost = costs[y * width + x];
                Vec2i best = {x, y};
                for (int i = 0; i < 4; i++) {
                    int nx = x + dx[i], ny = y + dy[i];
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        if (costs[ny * width + nx] < minCost) {
                            minCost = costs[ny * width + nx];
                            best = {nx, ny};
                        }
                    }
                }
                Vec3 dir = Vec3{(float)(best.x - x), 0, (float)(best.y - y)}.Normalized();
                directions[y * width + x] = dir;
            }
        }
    }

    Vec3 GetDirection(Vec3 pos) const {
        int x = (int)(pos.x / cellSize);
        int y = (int)(pos.z / cellSize);
        x = std::max(0, std::min(width - 1, x));
        y = std::max(0, std::min(height - 1, y));
        return directions[y * width + x];
    }
};
```

## 6. 完整仿真系统

```cpp
class SwarmSimulation {
    std::vector<Boid> boids;
    SpatialHash spatialHash;
    std::vector<Obstacle> obstacles;
    FlowField* sharedFlowField = nullptr;

    // 全局参数
    float separationWeight = 1.5f;
    float alignmentWeight = 1.0f;
    float cohesionWeight = 1.0f;
    float avoidanceWeight = 2.0f;
    float flowFieldWeight = 1.0f;

public:
    void Init(int boidCount, Vec3 spawnCenter, float spawnRadius) {
        spatialHash = SpatialHash(50.0f); // 单元格大小=感知范围

        boids.resize(boidCount);
        for (int i = 0; i < boidCount; i++) {
            boids[i].position = spawnCenter + RandomInSphere(spawnRadius);
            boids[i].velocity = RandomDirection() * 5.0f;
            boids[i].maxSpeed = 10.0f;
            boids[i].maxForce = 3.0f;
            boids[i].perceptionRadius = 50.0f;
        }
    }

    void Update(float dt) {
        // 重建空间哈希
        spatialHash.Clear();
        for (Boid& b : boids) spatialHash.Insert(&b);

        for (Boid& boid : boids) {
            // 获取邻居
            auto neighbors = spatialHash.Query(boid.position, boid.perceptionRadius);

            // 计算所有力
            Vec3 sep = Separation(boid, neighbors, 25.0f) * separationWeight;
            Vec3 ali = Alignment(boid, neighbors, boid.perceptionRadius) * alignmentWeight;
            Vec3 coh = Cohesion(boid, neighbors, boid.perceptionRadius) * cohesionWeight;
            Vec3 obs = AvoidObstacles(boid, obstacles) * avoidanceWeight;

            Vec3 totalForce = sep + ali + coh + obs;

            // 流场导航（如果有）
            if (sharedFlowField) {
                Vec3 flowDir = sharedFlowField->GetDirection(boid.position);
                Vec3 flowForce = (flowDir * boid.maxSpeed - boid.velocity) * flowFieldWeight;
                totalForce = totalForce + flowForce;
            }

            // 物理更新
            boid.acceleration = totalForce;
            boid.velocity = boid.velocity + boid.acceleration * dt;
            boid.velocity = LimitMagnitude(boid.velocity, boid.maxSpeed);
            boid.position = boid.position + boid.velocity * dt;
        }
    }

    const std::vector<Boid>& GetBoids() const { return boids; }
};
```

## 7. Unity实现示例

```csharp
using UnityEngine;
using System.Collections.Generic;

public class BoidController : MonoBehaviour {
    public int boidCount = 500;
    public GameObject boidPrefab;
    public float maxSpeed = 5f;
    public float maxForce = 2f;
    public float perceptionRadius = 3f;

    [Header("权重")]
    public float separationWeight = 1.5f;
    public float alignmentWeight = 1.0f;
    public float cohesionWeight = 1.0f;

    private List<BoidAgent> agents = new List<BoidAgent>();
    private Dictionary<Vector3Int, List<BoidAgent>> grid =
        new Dictionary<Vector3Int, List<BoidAgent>>();

    void Start() {
        for (int i = 0; i < boidCount; i++) {
            GameObject go = Instantiate(boidPrefab,
                Random.insideUnitSphere * 10f, Random.rotation);
            BoidAgent agent = go.GetComponent<BoidAgent>();
            agent.Init(maxSpeed, maxForce);
            agents.Add(agent);
        }
    }

    void Update() {
        // 构建空间哈希
        grid.Clear();
        float cellSize = perceptionRadius;
        foreach (var agent in agents) {
            Vector3Int key = new Vector3Int(
                Mathf.FloorToInt(agent.transform.position.x / cellSize),
                Mathf.FloorToInt(agent.transform.position.y / cellSize),
                Mathf.FloorToInt(agent.transform.position.z / cellSize));
            if (!grid.ContainsKey(key)) grid[key] = new List<BoidAgent>();
            grid[key].Add(agent);
        }

        // 每只Boid计算力
        foreach (var agent in agents) {
            Vector3 sep = Separation(agent) * separationWeight;
            Vector3 ali = Alignment(agent) * alignmentWeight;
            Vector3 coh = Cohesion(agent) * cohesionWeight;

            agent.ApplyForce(sep + ali + coh);
        }
    }

    Vector3 Separation(BoidAgent agent) {
        Vector3 steer = Vector3.zero;
        int count = 0;
        foreach (var other in GetNeighbors(agent, perceptionRadius * 0.5f)) {
            float dist = Vector3.Distance(agent.transform.position,
                                           other.transform.position);
            if (dist > 0 && dist < perceptionRadius * 0.5f) {
                Vector3 diff = (agent.transform.position -
                                other.transform.position).normalized / dist;
                steer += diff;
                count++;
            }
        }
        if (count > 0) steer /= count;
        if (steer.magnitude > 0) {
            steer = steer.normalized * maxSpeed - agent.velocity;
            steer = Vector3.ClampMagnitude(steer, maxForce);
        }
        return steer;
    }

    List<BoidAgent> GetNeighbors(BoidAgent agent, float radius) {
        List<BoidAgent> result = new List<BoidAgent>();
        Vector3Int center = new Vector3Int(
            Mathf.FloorToInt(agent.transform.position.x / radius),
            Mathf.FloorToInt(agent.transform.position.y / radius),
            Mathf.FloorToInt(agent.transform.position.z / radius));
        for (int x = -1; x <= 1; x++)
            for (int y = -1; y <= 1; y++)
                for (int z = -1; z <= 1; z++) {
                    Vector3Int key = center + new Vector3Int(x, y, z);
                    if (grid.ContainsKey(key)) result.AddRange(grid[key]);
                }
        return result;
    }

    // Alignment和Cohesion类似，省略...
    Vector3 Alignment(BoidAgent agent) { /* 类似实现 */ return Vector3.zero; }
    Vector3 Cohesion(BoidAgent agent) { /* 类似实现 */ return Vector3.zero; }
}

public class BoidAgent : MonoBehaviour {
    [HideInInspector] public Vector3 velocity;
    private float maxSpeed, maxForce;
    private Vector3 acc = Vector3.zero;

    public void Init(float ms, float mf) {
        maxSpeed = ms; maxForce = mf;
        velocity = Random.insideUnitSphere * maxSpeed;
    }

    public void ApplyForce(Vector3 force) { acc += force; }

    void Update() {
        velocity += acc * Time.deltaTime;
        velocity = Vector3.ClampMagnitude(velocity, maxSpeed);
        transform.position += velocity * Time.deltaTime;
        if (velocity.magnitude > 0.01f)
            transform.rotation = Quaternion.LookRotation(velocity);
        acc = Vector3.zero;
    }
}
```

## 8. RVO/ORCA算法

### 8.1 互易速度障碍

RVO（Reciprocal Velocity Obstacle）是一种数学上保证无碰撞的速度选择算法：

```
概念：
- 速度障碍（VO）：其他Agent的位置+速度在时间t内会"阻挡"的速度区域
- 互易（Reciprocal）：双方各承担一半避让责任
- ORCA：最优互Reciprocal碰撞避免，计算半平面约束
```

```cpp
// 简化的ORCA实现
struct ORCALine {
    Vec3 point;    // 半平面上的点
    Vec3 direction; // 半平面法线方向
};

std::vector<ORCALine> ComputeORCA(const Boid& agent,
                                    const std::vector<Boid*>& neighbors,
                                    float timeHorizon,
                                    float agentRadius) {
    std::vector<ORCALine> lines;

    for (const Boid* other : neighbors) {
        Vec3 relPos = other->position - agent.position;
        Vec3 relVel = agent.velocity - other->velocity;
        float distSq = relPos.LengthSq();
        float combinedRadius = agentRadius * 2.0f;
        float combinedRadiusSq = combinedRadius * combinedRadius;

        ORCALine line;

        if (distSq > combinedRadiusSq) {
            // 不在碰撞中：计算VO
            Vec3 w = relVel - relPos * (1.0f / timeHorizon);
            float wLenSq = w.LengthSq();

            float dotProduct = Dot(w, relPos);

            if (dotProduct < 0 && dotProduct * dotProduct > combinedRadiusSq * wLenSq) {
                // 速度在VO锥内
                float wLength = sqrtf(wLenSq);
                Vec3 unitW = w / wLength;
                line.direction = Vec3{unitW.z, 0, -unitW.x}; // 法线
                line.point = (agent.velocity + unitW * (combinedRadius / timeHorizon - wLength)) * 0.5f;
            } else {
                // 速度在VO锥外
                float legLength = sqrtf(distSq - combinedRadiusSq);
                Vec3 leftLegDir = {
                    relPos.x * legLength - relPos.z * combinedRadius,
                    0,
                    relPos.x * combinedRadius + relPos.z * legLength
                };
                // ... 计算ORCA半平面
            }
        } else {
            // 已在碰撞中：紧急避让
        }

        lines.push_back(line);
    }

    return lines;
}
```

## 9. 性能分析

### 9.1 时间复杂度

| 方法 | 每帧复杂度 | 最大Boid数（60fps） |
|------|-----------|-------------------|
| 暴力搜索 | O(N^2) | ~500 |
| 空间哈希 | O(N*k) | ~10,000 |
| 空间哈希+SIMD | O(N*k/4) | ~30,000 |
| GPU并行 | O(N*k/threads) | ~100,000+ |

k为平均邻居数（通常5-30）。

### 9.2 内存分析

| 组件 | 单位大小 |
|------|----------|
| Boid结构体 | 48-64字节 |
| 空间哈希条目 | 16字节（指针+键） |
| 流场格子 | 12-16字节（方向向量） |

## 10. 常见陷阱

1. **权重调优困难**：不同场景需要不同权重组合，缺乏系统方法。建议：使用调参工具实时调整
2. **局部振荡**：两只Boid反复互相避让形成死锁。解决方案：加入随机扰动
3. **穿越地形**：忽略地面约束导致Boid穿过地形。解决方案：射线投射约束高度
4. **性能瓶颈**：大量Boid时邻居搜索成瓶颈。必须使用空间分区
5. **视觉穿模**：Boid间视觉穿模。解决方案：使用渲染LOD+简化的碰撞体
6. **缺少全局协调**：群体行为过度随机，缺乏目的性。解决方案：加入领导者或流场引导

## 11. 实际游戏案例

### 案例1：孢子（Spore）

孢中的生物群体行为是Boids算法的经典应用：
- 使用Boids三规则驱动鱼群、鸟群、兽群
- 不同生物类型有不同权重（鱼群聚集性强，鸟群对齐性强）
- 群体规模从数十到数百不等
- 使用空间分区优化性能

### 案例2：全面战争系列

大规模士兵阵型使用群体AI：
- 士兵按阵型排列（方阵、线列、散兵）
- 编队系统控制群体移动方向
- RVO处理士兵间的碰撞避免
- 士兵个体有简化的Boids行为（跟随、分离）
- 使用LOD：远处士兵简化为群体粒子

### 案例3：僵尸围城类游戏

僵尸群的蜂拥行为：
- 简化版Boids：只有分离+聚集（无对齐）
- 大量僵尸使用流场导航到同一目标
- 僵尸个体碰撞使用简单的排斥力
- 使用GPU计算支持数万僵尸

### 案例4：蝙蝠侠阿卡姆系列

蝙蝠群的视觉效果：
- 使用Boids算法驱动蝙蝠群的飞行
- 蝙蝠群围绕蝙蝠侠飞行形成漩涡效果
- 渲染使用GPU Instancing支持数千只蝙蝠
- 物理使用简化碰撞避免
