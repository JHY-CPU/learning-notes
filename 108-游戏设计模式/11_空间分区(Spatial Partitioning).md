# 空间分区（Spatial Partitioning）

## 核心概念

空间分区将游戏世界划分为若干区域，加速空间查询（碰撞检测、可见性判断、最近目标查找）。它的核心价值是将 O(n^2) 的全局两两比较降为 O(n log n) 或 O(k) 的局部搜索（k 为实际邻近对象数）。

### 为什么需要空间分区？

```
没有空间分区时的碰撞检测：
foreach (objA in allObjects)        // O(n)
    foreach (objB in allObjects)    // O(n)
        if (objA != objB && Intersects(objA, objB))
            ResolveCollision();
// 总复杂度：O(n^2) —— 10000个对象需要1亿次比较！

有空间分区时：
foreach (objA in allObjects)        // O(n)
    var nearby = spatialIndex.Query(objA.bounds);  // O(log n) 或 O(k)
    foreach (objB in nearby)        // O(k)，k远小于n
        if (Intersects(objA, objB))
            ResolveCollision();
// 总复杂度：O(n * log n) 或 O(n * k)
```

### 四叉树（Quadtree）完整实现

四叉树递归将2D空间四等分，直到每个区域内的对象数量低于阈值。

```csharp
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// 四叉树——2D空间分区数据结构
/// </summary>
public class Quadtree
{
    private const int MAX_OBJECTS = 4;   // 每个节点最多容纳对象数
    private const int MAX_LEVELS = 6;    // 最大递归深度

    private int level;
    private List<QuadtreeObject> objects = new();
    private Rect bounds;
    private Quadtree[] children; // null = 叶节点

    public Quadtree(int level, Rect bounds)
    {
        this.level = level;
        this.bounds = bounds;
    }

    /// <summary>
    /// 插入对象
    /// </summary>
    public void Insert(GameObject obj)
    {
        // 如果有子节点，尝试放入子节点
        if (children != null)
        {
            int index = GetChildIndex(obj.Bounds);
            if (index != -1)
            {
                children[index].Insert(obj);
                return;
            }
            // 跨越多个象限的对象放在当前节点
        }

        objects.Add(new QuadtreeObject { Obj = obj, Bounds = obj.Bounds });

        // 超过阈值且未到最大层级 → 分裂
        if (objects.Count > MAX_OBJECTS && level < MAX_LEVELS)
        {
            Split();

            // 重新分配已有对象
            var allObjects = objects.ToList();
            objects.Clear();

            foreach (var qObj in allObjects)
            {
                int index = GetChildIndex(qObj.Bounds);
                if (index != -1)
                    children[index].Insert(qObj.Obj);
                else
                    objects.Add(qObj); // 跨象限的留在父节点
            }
        }
    }

    /// <summary>
    /// 移除对象
    /// </summary>
    public bool Remove(GameObject obj)
    {
        // 先在当前节点查找
        int index = objects.FindIndex(o => o.Obj == obj);
        if (index != -1)
        {
            objects.RemoveAt(index);
            TryMerge(); // 尝试合并（收缩）
            return true;
        }

        // 在子节点中查找
        if (children != null)
        {
            int childIndex = GetChildIndex(obj.Bounds);
            if (childIndex != -1)
            {
                bool removed = children[childIndex].Remove(obj);
                if (removed) TryMerge();
                return removed;
            }
        }

        return false;
    }

    /// <summary>
    /// 查询区域内的所有对象
    /// </summary>
    public List<GameObject> Query(Rect area)
    {
        var result = new List<GameObject>();

        // 当前节点的对象
        foreach (var qObj in objects)
        {
            if (area.Overlaps(qObj.Bounds))
                result.Add(qObj.Obj);
        }

        // 递归查询子节点
        if (children != null)
        {
            for (int i = 0; i < 4; i++)
            {
                if (children[i].bounds.Overlaps(area))
                    result.AddRange(children[i].Query(area));
            }
        }

        return result;
    }

    /// <summary>
    /// 查询最近的N个对象
    /// </summary>
    public List<GameObject> QueryNearest(Vector2 point, int maxCount, float maxRadius)
    {
        var candidates = Query(Rect.MinMaxRect(
            point.x - maxRadius, point.y - maxRadius,
            point.x + maxRadius, point.y + maxRadius));

        return candidates
            .OrderBy(obj => Vector2.Distance(obj.Position2D, point))
            .Take(maxCount)
            .ToList();
    }

    /// <summary>
    /// 更新移动对象的位置
    /// </summary>
    public void Update(GameObject obj, Rect newBounds)
    {
        // 简单实现：先移除再插入
        Remove(obj);
        obj.Bounds = newBounds;
        Insert(obj);
    }

    /// <summary>
    /// 分裂当前节点为四个子节点
    /// </summary>
    private void Split()
    {
        float halfW = bounds.width / 2f;
        float halfH = bounds.height / 2f;
        float x = bounds.x;
        float y = bounds.y;

        children = new Quadtree[4];
        children[0] = new Quadtree(level + 1, new Rect(x, y, halfW, halfH));               // 左下
        children[1] = new Quadtree(level + 1, new Rect(x + halfW, y, halfW, halfH));       // 右下
        children[2] = new Quadtree(level + 1, new Rect(x, y + halfH, halfW, halfH));       // 左上
        children[3] = new Quadtree(level + 1, new Rect(x + halfW, y + halfH, halfW, halfH)); // 右上
    }

    /// <summary>
    /// 判断对象完全位于哪个子象限
    /// 返回-1表示对象跨越多个象限
    /// </summary>
    private int GetChildIndex(Rect rect)
    {
        float midX = bounds.x + bounds.width / 2f;
        float midY = bounds.y + bounds.height / 2f;

        bool top = rect.yMin >= midY;
        bool bottom = rect.yMax <= midY;
        bool left = rect.xMax <= midX;
        bool right = rect.xMin >= midX;

        if (top && left) return 2;
        if (top && right) return 3;
        if (bottom && left) return 0;
        if (bottom && right) return 1;

        return -1; // 跨越多个象限
    }

    /// <summary>
    /// 尝试合并子节点（当对象数低于阈值时收缩）
    /// </summary>
    private void TryMerge()
    {
        if (children == null) return;

        int totalObjects = objects.Count;
        for (int i = 0; i < 4; i++)
            totalObjects += children[i].GetAllObjects().Count;

        if (totalObjects <= MAX_OBJECTS)
        {
            // 合并所有子节点对象到当前节点
            for (int i = 0; i < 4; i++)
                objects.AddRange(children[i].objects);

            children = null; // 销毁子节点
        }
    }

    /// <summary>
    /// 获取所有对象（调试用）
    /// </summary>
    public List<GameObject> GetAllObjects()
    {
        var result = new List<GameObject>(objects.Select(o => o.Obj));
        if (children != null)
            for (int i = 0; i < 4; i++)
                result.AddRange(children[i].GetAllObjects());
        return result;
    }

    private class QuadtreeObject
    {
        public GameObject Obj;
        public Rect Bounds;
    }
}
```

### BVH（包围体层次结构）实现

BVH 用包围盒层次组织对象，特别适合动态场景（物理引擎常用）。

```csharp
/// <summary>
/// BVH 节点——自顶向下构建的 AABB 层次树
/// </summary>
public class BVHNode
{
    public AABB Bounds;
    public BVHNode Left;
    public BVHNode Right;
    public GameObject Object; // 仅叶节点有
    public int ObjectCount;   // 子树中的对象总数

    /// <summary>
    /// 从对象列表构建 BVH 树
    /// </summary>
    public static BVHNode Build(List<GameObject> objects, int start, int end)
    {
        if (start >= end) return null;

        var node = new BVHNode();

        if (end - start == 1)
        {
            // 叶节点
            node.Object = objects[start];
            node.Bounds = objects[start].Bounds;
            node.ObjectCount = 1;
            return node;
        }

        // 计算所有对象的总包围盒
        node.Bounds = ComputeBounds(objects, start, end);

        // 选择最长的轴进行分割
        Vector3 size = node.Bounds.Max - node.Bounds.Min;
        int axis = 0;
        if (size.y > size.x) axis = 1;
        if (size.z > (&size.x)[axis]) axis = 2;

        // 按选中轴的中心位置排序
        int mid = (start + end) / 2;
        SortByAxis(objects, start, end, axis);

        // 递归构建
        node.Left = Build(objects, start, mid);
        node.Right = Build(objects, mid, end);
        node.ObjectCount = (node.Left?.ObjectCount ?? 0) + (node.Right?.ObjectCount ?? 0);

        return node;
    }

    /// <summary>
    /// 查询与给定 AABB 相交的所有对象
    /// </summary>
    public void Query(AABB area, List<GameObject> results)
    {
        if (!Bounds.Intersects(area)) return; // 快速剪枝

        if (Object != null) // 叶节点
        {
            if (Object.Bounds.Intersects(area))
                results.Add(Object);
        }
        else
        {
            Left?.Query(area, results);
            Right?.Query(area, results);
        }
    }

    /// <summary>
    /// 射线查询
    /// </summary>
    public void Raycast(Ray ray, float maxDist, List<RaycastHit> results)
    {
        if (!Bounds.IntersectsRay(ray, maxDist)) return;

        if (Object != null)
        {
            if (Object.Bounds.IntersectsRay(ray, maxDist))
                results.Add(new RaycastHit { Object = Object, Distance = 0 });
        }
        else
        {
            Left?.Raycast(ray, maxDist, results);
            Right?.Raycast(ray, maxDist, results);
        }
    }

    private static AABB ComputeBounds(List<GameObject> objects, int start, int end)
    {
        var bounds = objects[start].Bounds;
        for (int i = start + 1; i < end; i++)
            bounds = AABB.Union(bounds, objects[i].Bounds);
        return bounds;
    }

    private static void SortByAxis(List<GameObject> objects, int start, int end, int axis)
    {
        objects.Sort(start, end - start, Comparer<GameObject>.Create((a, b) =>
        {
            float ca = (&a.Bounds.Min.x)[axis] + (&a.Bounds.Max.x)[axis];
            float cb = (&b.Bounds.Min.x)[axis] + (&b.Bounds.Max.x)[axis];
            return ca.CompareTo(cb);
        }));
    }
}

public struct AABB
{
    public Vector3 Min;
    public Vector3 Max;

    public bool Intersects(AABB other)
    {
        return Min.x <= other.Max.x && Max.x >= other.Min.x &&
               Min.y <= other.Max.y && Max.y >= other.Min.y &&
               Min.z <= other.Max.z && Max.z >= other.Min.z;
    }

    public bool IntersectsRay(Ray ray, float maxDist)
    {
        // Slab 方法（见射线检测章节）
        return true; // 简化
    }

    public static AABB Union(AABB a, AABB b)
    {
        return new AABB
        {
            Min = Vector3.Min(a.Min, b.Min),
            Max = Vector3.Max(a.Max, b.Max)
        };
    }

    public Vector3 Center => (Min + Max) * 0.5f;
}
```

### 均匀网格（Spatial Hash Grid）

最简单的空间分区——将空间均匀划分为网格。

```csharp
/// <summary>
/// 空间哈希网格——O(1) 插入/查询，适合对象大小相近的场景
/// </summary>
public class SpatialGrid
{
    private readonly float cellSize;
    private readonly Dictionary<long, List<GameObject>> cells = new();

    public SpatialGrid(float cellSize)
    {
        this.cellSize = cellSize;
    }

    /// <summary>
    /// 将世界坐标转换为网格坐标
    /// </summary>
    private long HashPosition(float x, float y)
    {
        int cx = Mathf.FloorToInt(x / cellSize);
        int cy = Mathf.FloorToInt(y / cellSize);
        // 组合两个int为一个long作为哈希键
        return ((long)cx << 32) | (uint)cy;
    }

    public void Insert(GameObject obj)
    {
        // 对象可能跨越多个格子
        int minCx = Mathf.FloorToInt(obj.Bounds.xMin / cellSize);
        int minCy = Mathf.FloorToInt(obj.Bounds.yMin / cellSize);
        int maxCx = Mathf.FloorToInt(obj.Bounds.xMax / cellSize);
        int maxCy = Mathf.FloorToInt(obj.Bounds.yMax / cellSize);

        for (int cx = minCx; cx <= maxCx; cx++)
        {
            for (int cy = minCy; cy <= maxCy; cy++)
            {
                long key = ((long)cx << 32) | (uint)cy;
                if (!cells.ContainsKey(key))
                    cells[key] = new List<GameObject>();
                cells[key].Add(obj);
            }
        }
    }

    public void Remove(GameObject obj)
    {
        int minCx = Mathf.FloorToInt(obj.Bounds.xMin / cellSize);
        int minCy = Mathf.FloorToInt(obj.Bounds.yMin / cellSize);
        int maxCx = Mathf.FloorToInt(obj.Bounds.xMax / cellSize);
        int maxCy = Mathf.FloorToInt(obj.Bounds.yMax / cellSize);

        for (int cx = minCx; cx <= maxCx; cx++)
            for (int cy = minCy; cy <= maxCy; cy++)
            {
                long key = ((long)cx << 32) | (uint)cy;
                cells[key]?.Remove(obj);
            }
    }

    /// <summary>
    /// 查询区域内的对象
    /// </summary>
    public HashSet<GameObject> Query(Rect area)
    {
        var result = new HashSet<GameObject>();

        int minCx = Mathf.FloorToInt(area.xMin / cellSize);
        int minCy = Mathf.FloorToInt(area.yMin / cellSize);
        int maxCx = Mathf.FloorToInt(area.xMax / cellSize);
        int maxCy = Mathf.FloorToInt(area.yMax / cellSize);

        for (int cx = minCx; cx <= maxCx; cx++)
        {
            for (int cy = minCy; cy <= maxCy; cy++)
            {
                long key = ((long)cx << 32) | (uint)cy;
                if (cells.TryGetValue(key, out var cellObjects))
                {
                    foreach (var obj in cellObjects)
                    {
                        if (area.Overlaps(obj.Bounds))
                            result.Add(obj);
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// 清空网格（每帧重建或标记清除）
    /// </summary>
    public void Clear() => cells.Clear();
}
```

### 碰撞检测的粗检测阶段

```csharp
/// <summary>
/// 碰撞系统——使用空间分区进行粗检测（Broad Phase）
/// </summary>
public class CollisionSystem
{
    private Quadtree spatialIndex;
    private List<GameObject> allObjects = new();

    /// <summary>
    /// 粗检测（Broad Phase）：快速排除不可能碰撞的对象对
    /// </summary>
    public List<(GameObject, GameObject)> BroadPhase()
    {
        var potentialPairs = new HashSet<(GameObject, GameObject)>();

        foreach (var obj in allObjects)
        {
            if (!obj.HasCollider) continue;

            // 查询附近对象（而非全局两两比较）
            var nearby = spatialIndex.Query(obj.Bounds.Expand(collisionMargin));

            foreach (var other in nearby)
            {
                if (other == obj || !other.HasCollider) continue;
                // 避免重复对：只保留 ID 较小在前的对
                if (obj.ID < other.ID)
                    potentialPairs.Add((obj, other));
            }
        }

        return potentialPairs.ToList();
    }

    /// <summary>
    /// 细检测（Narrow Phase）：精确碰撞检测
    /// </summary>
    public void NarrowPhase(List<(GameObject, GameObject)> candidates)
    {
        foreach (var (a, b) in candidates)
        {
            // 根据碰撞体类型选择精确检测算法
            if (GJK.Detect(a.Collider, b.Collider, out var contact))
            {
                // 生成碰撞响应
                ResolveCollision(a, b, contact);
            }
        }
    }

    /// <summary>
    /// 每帧更新：更新空间分区 + 碰撞检测
    /// </summary>
    public void Update()
    {
        // 1. 更新空间分区中的对象位置
        spatialIndex = new Quadtree(0, worldBounds);
        foreach (var obj in allObjects)
            spatialIndex.Insert(obj);

        // 2. 粗检测
        var candidates = BroadPhase();

        // 3. 细检测
        NarrowPhase(candidates);
    }

    private void ResolveCollision(GameObject a, GameObject b, ContactInfo contact) { /* ... */ }
}
```

## 方案对比

| 方案 | 插入复杂度 | 查询复杂度 | 动态对象支持 | 内存开销 | 适用场景 |
|------|-----------|-----------|------------|---------|---------|
| 暴力比较 | O(1) | O(n^2) | 好 | 无 | <100个对象 |
| 均匀网格 | O(1) | O(k) | 一般 | 中 | 对象大小相近 |
| 四叉树/八叉树 | O(log n) | O(log n + k) | 一般 | 中 | 2D/3D通用 |
| BVH | O(n log n) | O(log n + k) | 优秀 | 中 | 物理引擎 |
| k-d树 | O(n log n) | O(log n + k) | 差 | 低 | 最近邻查询 |

## 常见陷阱与解决方案

1. **动态对象频繁重构**：大量移动对象导致树频繁分裂/合并。解决方案：使用松散四叉树（Loose Quadtree）或每帧重建
2. **对象大小差异大**：大对象跨越多个格子。解决方案：将大对象放在父节点，或使用 BVH
3. **查询边界效应**：查询区域边缘的对象可能被遗漏。解决方案：查询区域扩展一个安全边距
4. **cellSize 选择不当**：太小导致对象跨多格，太大导致每格对象太多。解决方案：设为平均对象大小的 2-5 倍

## Unity 实现

```csharp
// Unity 的 Physics 使用 PhysX 引擎，内部使用 BVH 进行粗检测
void Update()
{
    // Physics.OverlapSphere 内部使用空间分区加速
    var colliders = Physics.OverlapSphere(transform.position, 10f);
    foreach (var col in colliders)
    {
        // 处理范围内的对象
    }

    // Physics.Raycast 同样利用空间分区
    if (Physics.Raycast(ray, out RaycastHit hit, 100f))
    {
        // 命中检测
    }
}
```

## Unreal Engine 实现

```cpp
// UE 的可见性系统使用八叉树 + BSP 进行场景剔除
// UE 的 World Partition 使用空间分区管理大世界加载

// 自定义空间分区
class FSpatialHash
{
    TMap<int64, TArray<AActor*>> Grid;
    float CellSize = 1000.f;

    int64 HashLocation(FVector Location) const
    {
        int32 X = FMath::FloorToInt(Location.X / CellSize);
        int32 Y = FMath::FloorToInt(Location.Y / CellSize);
        return ((int64)X << 32) | (uint32)Y;
    }

    TArray<AActor*> QueryRadius(FVector Center, float Radius)
    {
        TArray<AActor*> Result;
        // 查询所有在半径范围内的格子
        // ...
        return Result;
    }
};
```

## 实际使用案例

- **Unity 的 PhysX** 使用 BVH 进行碰撞粗检测
- **《我的世界》** 使用均匀网格（Chunk）管理方块
- **《英雄联盟》** 的战争迷雾和技能碰撞使用四叉树加速
- **Unreal Engine** 的可见性系统使用八叉树 + BSP 进行场景剔除
- **《星际争霸2》** 使用均匀网格管理单位的空间查询
- **《GTA5》** 使用多层空间分区管理巨大的开放世界
