# 遮挡剔除(Occlusion Culling)

## 核心概念

遮挡剔除用于跳过被其他物体完全遮挡的渲染对象，减少不必要的Draw Call和Overdraw。在一个复杂的城市场景中，玩家可能只能看到20%的物体，如果不做遮挡剔除，GPU会浪费大量时间渲染被建筑物遮挡的物体。

### 可见性剔除的三个层次

1. **视锥剔除(Frustum Culling)**：剔除视锥体外的物体，引擎默认执行，零配置开销
2. **遮挡剔除(Occlusion Culling)**：剔除被其他物体完全遮挡的物体，需要额外配置
3. **背面剔除(Back-face Culling)**：GPU硬件阶段跳过朝向背离相机的面，自动执行

### PVS（Potentially Visible Set）预计算方案

PVS是Unity和UE中常用的遮挡剔除方案：
- 编辑器中预计算离散采样点（Cell）的可见性数据
- 运行时根据相机位置查找最近的Cell，获取该位置的可见物体列表
- 优点：运行时零CPU开销，查找速度快
- 缺点：需要烘焙时间，占用磁盘空间，不适合大规模动态场景

### 实时遮挡剔除方案

- **硬件遮挡查询(Hardware Occlusion Query)**：GPU判断物体是否可见，精度高但有延迟
- **软件光栅化遮挡**：用CPU低分辨率光栅化遮挡体，UE5的Nanite使用此方案
- **层次Z-Buffer(Hi-Z)**：每级Mipmap记录最大深度，快速判断遮挡

## 具体实现方法

### Unity Occlusion Culling完整设置

```
步骤1: 标记物体
- 将大型遮挡物标记为Occluder Static（墙壁、建筑、地形）
- 将可能被遮挡的物体标记为Occludee Static（道具、NPC、装饰）
- 注意：只有Static物体参与预计算

步骤2: 配置参数 (Window > Rendering > Occlusion Culling)
- Smallest Occluder: 最小遮挡物尺寸（默认5m）
  低于此值的物体不作为遮挡物。值越小烘焙数据越大
- Smallest Hole: 最小孔洞尺寸（默认0.25m）
  小于此值的孔洞视为封闭
- Backface Threshold: 背面阈值（默认100%）
  背面比例超过此值的Cell不烘焙

步骤3: 烘焙
- 点击Bake按钮
- 复杂场景可能需要几分钟到几小时
- 烘焙数据保存在OcclusionCullingData文件夹

步骤4: 验证
- 在Occlusion窗口切换到Visualization模式
- 移动Scene视图相机查看剔除效果
- 绿色=可见，灰色=被遮挡
```

### Unity动态遮挡剔除

```csharp
using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 动态物体的遮挡剔除
/// PVS只对静态物体有效，动态物体需要运行时方案
/// </summary>
public class DynamicOcclusionCuller : MonoBehaviour
{
    [Header("配置")]
    [SerializeField] private Camera mainCamera;
    [SerializeField] private float checkInterval = 0.2f; // 检测间隔（秒）
    [SerializeField] private int maxRaycastsPerFrame = 20; // 每帧最大射线数
    [SerializeField] private LayerMask occluderLayer;

    private List<Renderer> managedRenderers = new List<Renderer>();
    private float checkTimer;
    private int currentCheckIndex;

    public void Register(Renderer renderer)
    {
        if (!managedRenderers.Contains(renderer))
            managedRenderers.Add(renderer);
    }

    public void Unregister(Renderer renderer)
    {
        managedRenderers.Remove(renderer);
    }

    void Update()
    {
        checkTimer += Time.deltaTime;
        if (checkTimer < checkInterval) return;
        checkTimer = 0;

        // 分帧检测：每帧只检测一部分物体
        int checkCount = Mathf.Min(maxRaycastsPerFrame, managedRenderers.Count);
        for (int i = 0; i < checkCount; i++)
        {
            int index = (currentCheckIndex + i) % managedRenderers.Count;
            CheckVisibility(managedRenderers[index]);
        }
        currentCheckIndex = (currentCheckIndex + checkCount) % managedRenderers.Count;
    }

    void CheckVisibility(Renderer renderer)
    {
        if (renderer == null) return;

        // 第一步：视锥剔除（快速）
        if (!GeometryUtility.TestPlanesAABB(
            GeometryUtility.CalculateFrustumPlanes(mainCamera), renderer.bounds))
        {
            renderer.enabled = false;
            return;
        }

        // 第二步：遮挡剔除（射线检测）
        Vector3 cameraPos = mainCamera.transform.position;
        Vector3 targetPos = renderer.bounds.center;
        Vector3 direction = targetPos - cameraPos;
        float distance = direction.magnitude;

        // 从相机到物体中心发射射线
        bool occluded = Physics.Raycast(
            cameraPos, direction.normalized,
            distance, occluderLayer);

        renderer.enabled = !occluded;
    }
}
```

### UE Precomputed Visibility配置

```
步骤:
1. 在关卡中放置Precomputed Visibility Volume
   - 覆盖玩家所有可能到达的区域
   - 不要覆盖玩家无法到达的区域（节省数据量）

2. World Settings中配置:
   - Precomputed Visibility Cell Size: 200 (默认)
     值越小精度越高但数据量越大
   - Precomputed Visibility Cell Height: 200

3. 构建光照时自动计算可见性数据

4. 运行时调试命令:
   - r.ShowPrecomputedVisibility 1  // 可视化显示Cell
   - stat initviews  // 查看剔除统计
   - r.VisualizeOccludedPrimitives 1  // 显示被剔除的物体
```

### UE HLOD与软件剔除

```
Hierarchical LOD (HLOD):
- 将远处多个小物体的网格合并为一个简化模型
- 大幅减少远处场景的Draw Call
- 在World Partition中自动管理大规模世界的可见性

配置步骤:
1. 选择要合并的Actor
2. 右键 > Level Actor > Create HLOD
3. 设置HLOD层级距离
4. Build HLOD Mesh

HLOD与遮挡剔除配合：
- 近处：PVS预计算剔除
- 中距离：视锥剔除 + 距离剔除
- 远距离：HLOD合并 + 视锥剔除
```

## 性能基准数据

| 场景 | 无遮挡剔除 | PVS预计算 | 动态射线检测 |
|------|-----------|----------|------------|
| 室内1000物体 | 800 DC, 5ms | 200 DC, 1.2ms | 300 DC, 2ms |
| 城市5000物体 | 3000 DC, 12ms | 800 DC, 3ms | 1200 DC, 5ms |
| 开放世界 | 视锥剔除足够 | 数据量过大 | 射线开销大 |
| CPU开销 | 0 | <0.1ms | 1-3ms |

## 最佳实践

- 室内场景优先使用Occlusion Culling预计算方案，效果最好
- 开放世界使用视锥剔除 + 距离剔除组合，PVS数据量不可接受
- 将Occluder Static标记给大型墙壁、建筑、地形等遮挡物
- 小型物体标记为Occludee Static而非Occluder（减少烘焙数据）
- 定期用Occlusion可视化模式验证剔除效果
- 动态遮挡检测使用分帧检测，避免单帧射线过多
- 对远处物体使用距离剔除而非遮挡剔除（更高效）

## 常见陷阱与修复

**陷阱1：忘记标记静态遮挡物**
- 症状：预计算结果不准确，很多本应被遮挡的物体仍然可见
- 修复：确保大型墙壁和建筑标记为Occluder Static

**陷阱2：Occlusion Culling烘焙数据占用过多磁盘空间**
- 症状：复杂场景的烘焙数据达数百MB
- 修复：增大Smallest Occluder值（减少小物体作为遮挡物），减少Cell密度

**陷阱3：动态物体不参与预计算遮挡剔除**
- 症状：移动的门/箱子后面的物体仍然被渲染
- 修复：为动态物体使用运行时遮挡检测方案

**陷阱4：Cell Size设置过大导致小空间内剔除不准确**
- 症状：小房间内物体被错误剔除或未被剔除
- 修复：减小Cell Size，增加采样密度

**陷阱5：移动平台上预计算数据加载增加启动时间**
- 症状：游戏启动加载时间增加2-5秒
- 修复：压缩Occlusion数据，或对移动端使用简化的遮挡方案
