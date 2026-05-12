# 导航网格（NavMesh）

## 1. 核心理论

### 1.1 什么是导航网格

导航网格（Navigation Mesh，简称NavMesh）是将游戏世界中的可行走区域表示为凸多边形网格的数据结构。与基于规则网格（Grid）的寻路不同，NavMesh直接在几何层面描述可行走空间，具有以下核心优势：

- **精度与效率平衡**：多边形数量远少于等效网格，寻路搜索空间小
- **地形适应**：天然支持斜坡、楼梯、不规则地形等复杂几何
- **内存高效**：存储多边形而非逐格信息，内存占用可降至网格的1/10
- **自然路径**：寻路结果沿地形表面移动，无需额外平滑处理

NavMesh最早由Monolith Productions在2003年的F.E.A.R.中系统性地提出和使用，后被Recast/Detour开源库推广，现已成为3D游戏寻路的事实标准。

### 1.2 凸多边形分解原理

NavMesh的核心是将可行走区域分解为**凸多边形**。凸多边形的关键性质：

- 任意两点之间的线段完全位于多边形内部
- 因此，在同一个凸多边形内，Agent可以直线行走无需额外避障
- 多边形之间的邻接关系构成一个图结构，用于路径搜索

**凸多边形分解的数学基础**：

给定一个由轮廓定义的多边形区域，分解为凸多边形的经典方法：
1. **三角化**：将多边形分解为三角形（每个三角形都是凸多边形）
2. **多边形合并**：合并相邻三角形为更大的凸多边形，减少节点数量
3. **约束优化**：保证合并后的多边形仍为凸，且边数不超过上限

### 1.3 寻路原理

NavMesh上的寻路分为两个阶段：

**阶段1：多边形图搜索**
- 节点：NavMesh中的多边形
- 边：多边形之间的共享边
- 算法：A*搜索，代价为多边形中心距离或共享边中点距离

**阶段2：漏斗算法（Funnel Algorithm）**
- 在多边形路径确定后，找到穿越各多边形共享边的最优通道
- 使用漏斗（Funnel）追踪通道的左右边界，收紧漏斗找到转弯点
- 输出沿通道中心的最优路径点序列

## 2. Recast库详解

Recast是业界最流行的开源NavMesh生成库，由Mikko Mononen开发。它将3D场景几何体转换为NavMesh数据。

### 2.1 Recast处理流程

```
输入几何体（OBJ/FBX顶点+索引）
    |
    v
① 体素化（Voxelization）
  将场景几何光栅化为体素柱（高度场）
  |
    v
② 过滤可行走表面（Filter Walkable Surfaces）
  标记坡度过大的表面为不可行走
  |
    v
③ 简化轮廓（Simplify Contours）
  使用Marching Squares提取等值线，合并相邻可行走区域
  |
    v
④ 构建多边形网格（Build Poly Mesh）
  将轮廓三角化，合并为凸多边形
  |
    v
⑤ 构建细节网格（Build Detail Mesh）
  为每个多边形生成高度细节，处理地形高度变化
  |
    v
输出：dtNavMesh数据结构
```

### 2.2 体素化参数详解

```cpp
// Recast关键配置参数
struct RecastConfig {
    float cellSize = 0.3f;      // XZ平面体素大小（越小精度越高，内存越大）
    float cellHeight = 0.2f;    // Y轴体素高度（影响高度精度）
    float agentHeight = 2.0f;   // Agent高度（用于过滤低矮空间）
    float agentRadius = 0.6f;   // Agent半径（用于膨胀障碍物）
    float agentMaxClimb = 0.9f; // 最大可攀爬高度
    float agentMaxSlope = 45.0f;// 最大可行走坡度（度数）
    int minRegionArea = 8;      // 最小区域面积（体素数）
    int mergeRegionArea = 20;   // 区域合并阈值
    float edgeMaxLen = 12.0f;   // 轮廓边最大长度
    float edgeMaxError = 1.3f;  // 轮廓简化最大误差
    int vertsPerPoly = 6;       // 每个多边形最大顶点数
    float detailSampleDist = 6.0f;  // 细节网格采样距离
    float detailSampleMaxError = 1.0f; // 细节采样最大误差
};
```

**参数影响分析**：

| 参数 | 减小值的效果 | 增大值的效果 |
|------|------------|------------|
| cellSize | 精度提高，内存增大，烘焙变慢 | 精度降低，内存减少，烘焙变快 |
| cellHeight | 高度精度提高 | 高度精度降低，悬崖处理变差 |
| agentRadius | 导航更灵活，但可能穿墙 | 安全距离增大，但窄道可能断开 |
| agentMaxSlope | 更多地形被标记为不可行走 | 更陡的坡可行走，但Agent可能滑落 |

### 2.3 Detour库：NavMesh查询

Detour是Recast的配套库，负责NavMesh的运行时查询。

```cpp
// Detour核心查询流程
class DetourNavMesh {
    dtNavMesh* m_navMesh;
    dtNavMeshQuery* m_navQuery;

public:
    bool Init(const unsigned char* navMeshData, int dataSize) {
        m_navMesh = dtAllocNavMesh();
        dtStatus status = m_navMesh->init(navMeshData, dataSize, DT_TILE_FREE_DATA);
        if (dtStatusFailed(status)) return false;

        m_navQuery = dtAllocNavMeshQuery();
        m_navQuery->init(m_navMesh, 2048); // 2048=最大搜索节点数
        return true;
    }

    // 寻路查询
    std::vector<dtPolyRef> FindPolyPath(float* startPos, float* endPos) {
        float extents[] = {2.0f, 4.0f, 2.0f}; // 搜索范围

        dtPolyRef startRef, endRef;
        float nearestStart[3], nearestEnd[3];

        // 将点投射到NavMesh
        m_navQuery->findNearestPoly(startPos, extents, &m_filter, &startRef, nearestStart);
        m_navQuery->findNearestPoly(endPos, extents, &m_filter, &endRef, nearestEnd);

        if (!startRef || !endRef) return {};

        // 多边形图A*搜索
        static const int MAX_POLYS = 2048;
        dtPolyRef polys[MAX_POLYS];
        int npolys;
        m_navQuery->findPath(startRef, endRef, nearestStart, nearestEnd,
                             &m_filter, polys, &npolys, MAX_POLYS);

        // 漏斗算法获取精确路径点
        float straightPath[MAX_POLYS * 3];
        unsigned char straightPathFlags[MAX_POLYS];
        dtPolyRef straightPathPolys[MAX_POLYS];
        int nstraightPath;
        m_navQuery->findStraightPath(nearestStart, nearestEnd, polys, npolys,
                                      straightPath, straightPathFlags,
                                      straightPathPolys, &nstraightPath, MAX_POLYS);

        std::vector<dtPolyRef> result;
        for (int i = 0; i < npolys; i++) result.push_back(polys[i]);
        return result;
    }
};
```

## 3. 漏斗算法（Funnel Algorithm）

### 3.1 算法原理

漏斗算法解决的问题是：给定一系列多边形路径和穿越共享边的通道，找到从起点到终点的最短路径点序列。

**核心思想**：
- 维护一个"漏斗"形状，由左顶点和右顶点定义
- 漏斗的顶点（apex）是当前确定的路径点
- 遍历通道序列，用新顶点更新漏斗的左右边界
- 当新顶点导致漏斗变窄到反转时，将相应边界顶点设为新路径点

### 3.2 完整实现

```cpp
struct FunnelResult {
    std::vector<Vec3> points;
};

FunnelResult FunnelAlgorithm(
    const std::vector<int>& polyPath,     // 多边形ID路径
    const NavMesh& navMesh,
    Vec3 startPos,
    Vec3 endPos)
{
    FunnelResult result;
    result.points.push_back(startPos);

    if (polyPath.size() <= 1) {
        result.points.push_back(endPos);
        return result;
    }

    Vec3 portalLeft[2048], portalRight[2048];
    int portalCount = 0;

    // 为每条共享边构建通道
    // 第一个通道是起点
    portalLeft[0] = startPos;
    portalRight[0] = startPos;
    portalCount = 1;

    for (size_t i = 0; i < polyPath.size() - 1; i++) {
        const NavPoly& polyA = navMesh.polys[polyPath[i]];
        const NavPoly& polyB = navMesh.polys[polyPath[i + 1]];

        // 找到共享边
        Vec3 edgeLeft, edgeRight;
        FindSharedEdge(polyA, polyB, edgeLeft, edgeRight);

        portalLeft[portalCount] = edgeLeft;
        portalRight[portalCount] = edgeRight;
        portalCount++;
    }

    // 最后一个通道是终点
    portalLeft[portalCount] = endPos;
    portalRight[portalCount] = endPos;
    portalCount++;

    // 漏斗扫描
    int apexIdx = 0;
    int leftIdx = 0;
    int rightIdx = 0;

    for (int i = 1; i < portalCount; i++) {
        // 更新右边界
        float cross = TriArea2D(portalLeft[apexIdx], portalRight[apexIdx], portalRight[i]);
        if (cross <= 0) {
            if (apexIdx == rightIdx || TriArea2D(portalLeft[apexIdx], portalLeft[rightIdx], portalRight[i]) > 0) {
                rightIdx = i;
            } else {
                // 收窄右边界，产生新路径点
                result.points.push_back(portalLeft[rightIdx]);
                apexIdx = rightIdx;
                i = apexIdx;
                leftIdx = apexIdx;
                rightIdx = apexIdx;
                continue;
            }
        }

        // 更新左边界
        cross = TriArea2D(portalLeft[apexIdx], portalRight[apexIdx], portalLeft[i]);
        if (cross >= 0) {
            if (apexIdx == leftIdx || TriArea2D(portalRight[apexIdx], portalRight[leftIdx], portalLeft[i]) < 0) {
                leftIdx = i;
            } else {
                result.points.push_back(portalRight[leftIdx]);
                apexIdx = leftIdx;
                i = apexIdx;
                leftIdx = apexIdx;
                rightIdx = apexIdx;
                continue;
            }
        }
    }

    result.points.push_back(endPos);
    return result;
}

// 计算三角形在XZ平面的有符号面积的2倍
float TriArea2D(const Vec3& a, const Vec3& b, const Vec3& c) {
    float ax = b.x - a.x, az = b.z - a.z;
    float bx = c.x - a.x, bz = c.z - a.z;
    return ax * bz - bx * az;
}
```

## 4. 动态障碍与运行时修改

### 4.1 动态障碍处理策略

| 方案 | 实现复杂度 | 适用场景 | 性能影响 |
|------|-----------|----------|----------|
| 局部重新烘焙 | 高 | 障碍频繁变化 | 中等 |
| 障碍标记 | 低 | 静态障碍物的开关 | 极低 |
| 动态NavMesh（tile系统） | 中 | 大世界的局部变化 | 低 |
| 导航修改器 | 中 | 区域属性变化 | 低 |
| 避障替代 | 低 | 临时障碍物 | 极低 |

### 4.2 Tile-based动态NavMesh

Detour支持将NavMesh划分为Tile网格，每个Tile独立烘焙：

```cpp
class DynamicNavMesh {
    dtNavMesh* m_navMesh;

public:
    // 更新单个Tile
    bool UpdateTile(int tileX, int tileZ, const RecastConfig& config,
                    const float* verts, int nverts,
                    const int* tris, int ntris)
    {
        // 1. 删除旧Tile
        dtTileRef oldTileRef = m_navMesh->getTileRefAt(tileX, tileZ, 0);
        if (oldTileRef) {
            m_navMesh->removeTile(oldTileRef, nullptr, nullptr);
        }

        // 2. 对该Tile区域重新烘焙NavMesh
        unsigned char* navMeshData = nullptr;
        int navMeshDataSize = 0;
        // ... 使用Recast对该区域重新生成navMeshData ...

        // 3. 添加新Tile
        dtStatus status = m_navMesh->addTile(navMeshData, navMeshDataSize,
                                              DT_TILE_FREE_DATA, 0, nullptr);
        return dtStatusSucceed(status);
    }
};
```

### 4.3 Off-Mesh Links（离网链接）

离网链接用于表示Agent可以跳跃、攀爬或通过非行走方式跨越的连接：

```cpp
// 在Recast中定义离网链接
struct OffMeshConnection {
    Vec3 startPos;    // 链接起点（在NavMesh上）
    Vec3 endPos;      // 链接终点（在NavMesh上）
    float radius;     // 链接半径（用于匹配Agent位置）
    int bidirectional; // 是否双向通行
    int areaType;     // 区域类型（跳跃/攀爬等）
    int flags;        // 通行标记
};

// 使用示例：跳跃链接
// 当Agent寻路到悬崖边缘时，如果存在off-mesh link到对面，
// 寻路器会将此链接加入路径，Agent执行跳跃动画到达对面
```

**应用场景**：
- 窗户翻越（如使命召唤中的战术移动）
- 梯子攀爬（如生化危机中的垂直移动）
- 平台跳跃（如古墓丽影中的攀爬系统）
- 电梯/传送门（空间扭曲连接）

## 5. 多Agent群体寻路

### 5.1 群体寻路挑战

当大量Agent使用NavMesh寻路时面临的问题：
- 多Agent重叠（多个Agent走同一路线）
- 通道阻塞（窄道上的Agent对冲）
- 路径震荡（Agent反复避让形成死锁）

### 5.2 解决方案

**方案1：路径规划层——局部避让**

```cpp
// ORCA（互Reciprocal Velocity Obstacle）算法
Vec3 ComputeORCAVelocity(Agent* agent, const std::vector<Agent*>& neighbors,
                           float timeHorizon) {
    Vec3 prefVel = agent->preferredVelocity;
    std::vector<HalfPlane> orcaLines;

    for (Agent* other : neighbors) {
        Vec3 relPos = other->pos - agent->pos;
        Vec3 relVel = agent->velocity - other->velocity;
        float distSq = relPos.LengthSq();
        float combinedRadius = agent->radius + other->radius;
        float combinedRadiusSq = combinedRadius * combinedRadius;

        if (distSq > combinedRadiusSq) {
            // 不在碰撞中：计算速度障碍锥
            Vec3 relPosNorm = relPos.Normalized();
            // ... 计算ORCA半平面 ...
            orcaLines.push_back(/* half plane */);
        } else {
            // 已在碰撞中：计算穿透恢复速度
        }
    }

    // 在所有ORCA半平面的交集中找到最接近期望速度的解
    return LinearProgramming(orcaLines, prefVel);
}
```

**方案2：流场共享**

对于同目标的大量Agent，使用共享流场：
- 预计算一次流场
- 所有Agent查询同一流场获取移动方向
- 每个Agent仅处理局部避障

**方案3：时间窗口调度**

将Agent的寻路请求分散到不同帧：
```cpp
class PathRequestManager {
    struct PathRequest {
        Agent* agent;
        Vec3 start, goal;
        int priority;
    };

    std::queue<PathRequest> requestQueue;
    int maxRequestsPerFrame = 10;

    void ProcessFrame() {
        int processed = 0;
        while (!requestQueue.empty() && processed < maxRequestsPerFrame) {
            auto req = requestQueue.front();
            requestQueue.pop();
            auto path = navMesh.FindPath(req.start, req.goal);
            req.agent->SetPath(path);
            processed++;
        }
    }
};
```

## 6. Unity中的NavMesh实现

### 6.1 Unity内置NavMesh系统

```csharp
using UnityEngine;
using UnityEngine.AI;

// Agent控制脚本
public class NavMeshAgentController : MonoBehaviour {
    private NavMeshAgent agent;
    public Transform target;

    void Start() {
        agent = GetComponent<NavMeshAgent>();
    }

    void Update() {
        if (target != null) {
            agent.SetDestination(target.position);
        }
    }

    // 动态修改NavMesh区域（如放置障碍物）
    public void AddObstacle(Vector3 center, Vector3 size) {
        NavMeshObstacle obstacle = gameObject.AddComponent<NavMeshObstacle>();
        obstacle.shape = NavMeshObstacleShape.Box;
        obstacle.center = center;
        obstacle.size = size;
        obstacle.carving = true;  // 实时从NavMesh挖洞
    }
}

// NavMesh烘焙设置
public class NavMeshBaker : MonoBehaviour {
    public void BakeNavMesh() {
        NavMeshSurface surface = GetComponent<NavMeshSurface>();
        surface.collectObjects = CollectObjects.All;
        surface.useGeometry = NavMeshCollectGeometry.PhysicsColliders;
        surface.buildHeightMesh = true;
        surface.agentTypeID = 0; // 默认Agent类型

        surface.BuildNavMesh();
    }
}
```

### 6.2 Off-Mesh Link使用

```csharp
// Unity中的跳跃链接
public class JumpLink : MonoBehaviour {
    public Transform startPoint;
    public Transform endPoint;
    public float jumpDuration = 1.0f;

    void Start() {
        OffMeshLink link = gameObject.AddComponent<OffMeshLink>();
        link.startTransform = startPoint;
        link.endTransform = endPoint;
        link.biDirectional = true;
        link.costOverride = 5.0f; // 跳跃代价高于普通行走
    }
}

// 处理Off-Mesh Link的Agent控制器
public class AgentLinkHandler : MonoBehaviour {
    private NavMeshAgent agent;

    IEnumerator Start() {
        agent = GetComponent<NavMeshAgent>();
        agent.autoTraverseOffMeshLink = false; // 手动处理

        while (true) {
            if (agent.isOnOffMeshLink) {
                yield return StartCoroutine(TraverseLink());
            }
            yield return null;
        }
    }

    IEnumerator TraverseLink() {
        OffMeshLinkData data = agent.currentOffMeshLinkData;
        Vector3 startPos = agent.transform.position;
        Vector3 endPos = data.endPos;

        float t = 0;
        while (t < 1.0f) {
            t += Time.deltaTime;
            // 抛物线跳跃
            Vector3 pos = Vector3.Lerp(startPos, endPos, t);
            pos.y += 4.0f * t * (1.0f - t); // 抛物线
            agent.transform.position = pos;
            yield return null;
        }

        agent.CompleteOffMeshLink();
    }
}
```

## 7. Unreal Engine中的NavMesh

UE使用Recast/Detour作为底层NavMesh系统，通过`UNavigationSystemV1`提供上层接口。

### 7.1 NavMesh配置

```cpp
// 在DefaultEngine.ini中配置
[/Script/NavigationSystem.RecastNavMesh]
AgentRadius=35.0
AgentHeight=144.0
AgentMaxSlope=44.0
AgentMaxStepHeight=45.0
CellSize=19.0
CellHeight=10.0
TileSizeUU=512.0
bDrawPolyEdges=True
bDistinctlyDrawTilesBeingBuilt=True
RuntimeGeneration=Dynamic  // 支持运行时动态更新
MaxSimplificationError=1.3
```

### 7.2 导航修饰器（NavModifierVolume）

```cpp
// 在蓝图或C++中使用NavModifierVolume修改导航属性
UCLASS()
class ANavModifierExample : public AActor {
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere)
    UNavModifierVolume* DangerZone; // 在编辑器中放置体积

    // 运行时动态修改区域属性
    void SetDangerZoneActive(bool active) {
        if (active) {
            DangerZone->SetAreaClass(UNavArea_Obstacle::StaticClass());
        } else {
            DangerZone->SetAreaClass(UNavArea_Default::StaticClass());
        }
    }
};

// 使用EQS（Environment Query System）与NavMesh配合
UCLASS()
class UEnvQueryTest_NavMesh : public UEnvQueryTest {
    GENERATED_BODY()

    virtual void RunTest(FEnvQueryInstance& QueryInstance) const override {
        // 测试候选点是否在NavMesh上可达
        UNavigationSystemV1* NavSys = UNavigationSystemV1::GetCurrent(GetWorld());
        for (int32 i = 0; i < QueryInstance.Items.Num(); i++) {
            FVector ItemLoc = QueryInstance.GetItemLocation(i);
            FNavLocation NavLoc;
            bool bOnNavMesh = NavSys->ProjectPointToNavigation(ItemLoc, NavLoc);
            QueryInstance.SetItemScore(i, bOnNavMesh ? 1.0f : 0.0f);
        }
    }
};
```

## 8. 性能分析与对比

### 8.1 NavMesh vs Grid vs Waypoint

| 维度 | Grid | NavMesh | Waypoint Graph |
|------|------|---------|----------------|
| 内存占用 | 高（O(W*H)） | 低（O(P)，P为多边形数） | 极低（O(N)） |
| 寻路精度 | 网格级 | 连续空间 | 依赖路点密度 |
| 地形适应 | 差（需要大量格子） | 好（天然贴合地形） | 差（仅路点附近） |
| 动态更新 | 修改格子值 | 重新烘焙区域 | 增删路点 |
| 路径质量 | 锯齿状，需平滑 | 自然流畅 | 依赖路点布局 |
| 3D支持 | 需要分层网格 | 原生支持 | 需要3D路点 |
| 实现复杂度 | 低 | 高 | 低 |

### 8.2 烘焙性能基准

典型场景（100m x 100m地形，中等复杂度建筑）的Recast烘焙时间：

| 配置 | 体素数 | 烘焙时间 | 内存占用 |
|------|--------|----------|----------|
| cellSize=0.3 | ~111K | ~50ms | ~2MB |
| cellSize=0.1 | ~1M | ~400ms | ~15MB |
| cellSize=0.05 | ~4M | ~2s | ~50MB |

## 9. 常见陷阱与调试

### 9.1 常见问题

1. **NavMesh断裂**：agentRadius过小或cellSize过大，导致窄道NavMesh不连续
2. **穿墙角**：漏斗算法生成的路径点过于贴近多边形边缘
3. **性能抖动**：动态重新烘焙导致单帧耗时突增（应使用异步烘焙）
4. **坡度误判**：cellHeight过大，导致陡坡被错误标记为可行走
5. **Agent浮空/沉入**：细节网格采样不足导致高度不精确

### 9.2 调试工具

- **RecastDemo**：开源可视化工具，可逐步查看烘焙各阶段
- **UE F11调试视图**：显示NavMesh多边形、路径、Agent位置
- **Unity NavMesh Display**：Window > AI > Navigation可视化
- **自定义热力图**：绘制Agent密度、路径热区

## 10. 实际游戏案例

### 案例1：使命召唤系列

使命召唤使用分Tile的NavMesh管理大型多人地图：
- 地图划分为512x512单位的Tile
- 每个Tile独立烘焙，支持运行时加载/卸载
- 使用NavMesh标记不同区域（可行走、不可行走、跳跃）
- 敌人AI通过EQS（Environment Query System）在NavMesh上选择最优掩护位置

### 案例2：刺客信条系列

刺客信条的城市环境需要处理极端垂直方向的NavMesh：
- 多层NavMesh：地面、屋顶、墙壁各一层
- 离网链接连接各层（梯子、跳跃点）
- 城市场景中NavMesh密度极高，使用LOD降级远处NavMesh精度

### 案例3：The Last of Us系列

The Last of Us的潜行AI深度依赖NavMesh：
- 使用NavMesh区域标记标记高警觉/低警觉区域
- 敌人巡逻路线沿NavMesh多边形规划
- 掩护点通过NavMesh边分析自动检测（凸多边形边缘对面有空间=掩护点）
- 动态障碍（如推倒的桌子）通过局部NavMesh更新处理
