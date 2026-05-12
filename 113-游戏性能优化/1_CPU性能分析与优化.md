# CPU性能分析与优化

## 核心概念

CPU性能是游戏帧率的主要瓶颈之一。在移动设备上，CPU往往比GPU更容易成为瓶颈，因为移动CPU核心频率低、缓存小、内存带宽有限。高效使用Profiler定位热点函数是性能优化的第一步，也是最重要的一步。

### Unity Profiler关键模块详解

- **CPU Usage**：按函数统计耗时，可按Hierarchy（调用树）和Timeline（时间线）两种视图查看。Hierarchy视图适合找热点函数，Timeline视图适合分析帧间模式
- **Memory**：追踪堆内存分配、GC触发和Asset引用。关注Mono Heap Size趋势，持续增长说明有内存泄漏
- **Rendering**：Draw Call、Batches、SetPass Calls、Shadow Casters等渲染指标
- **Physics**：物理模拟耗时、碰撞检测次数、射线检测次数
- **Audio**：音频DSP处理耗时、同时播放音源数量

### 热点函数定位方法论

1. **开启Deep Profile模式**：捕获完整调用栈（注意：Deep Profile本身有10-30%开销，仅在需要时开启）
2. **按Self Time排序**：Self Time高的函数是真正需要优化的。Total Time高但Self Time低说明时间花在子函数中
3. **关注GC Alloc列**：高分配量往往预示性能问题，即使当前没有GC暂停
4. **使用Hierarchy视图定位**：找到占用最高的父函数，逐层展开找到真正耗时的叶子函数
5. **对比不同帧**：找到卡顿帧和正常帧的差异

### 性能指标基准

| 平台 | 目标帧率 | 帧预算 | CPU预算 |
|------|---------|--------|--------|
| PC (60fps) | 60fps | 16.67ms | ~10ms |
| PC (144fps) | 144fps | 6.94ms | ~4ms |
| 移动端(60fps) | 60fps | 16.67ms | ~8ms |
| 移动端(30fps) | 30fps | 33.33ms | ~20ms |
| VR (90fps) | 90fps | 11.11ms | ~5ms |

## 具体实现方法

### Profiler代码标记（深度用法）

```csharp
using Unity.Profiling;

/// <summary>
/// 使用ProfilerMarker精确标记代码段
/// 在Profiler中可以看到自定义标记的耗时
/// </summary>
public class GameLogic : MonoBehaviour
{
    // 静态ProfilerMarker避免每帧GC分配
    static readonly ProfilerMarker s_UpdateAI =
        new ProfilerMarker("GameLogic.UpdateAI");
    static readonly ProfilerMarker s_Pathfinding =
        new ProfilerMarker(ProfilerCategory.Scripts, "Pathfinding");
    static readonly ProfilerMarker s_CombatLogic =
        new ProfilerMarker(ProfilerCategory.Scripts, "CombatLogic");

    // 带子标记的嵌套分析
    static readonly ProfilerMarker s_AI_Decision =
        new ProfilerMarker("AI.Decision");
    static readonly ProfilerMarker s_AI_Movement =
        new ProfilerMarker("AI.Movement");
    static readonly ProfilerMarker s_AI_Animation =
        new ProfilerMarker("AI.Animation");

    void Update()
    {
        using (s_UpdateAI.Auto()) // 使用using自动Begin/End
        {
            UpdateAllAI();
        }
    }

    void UpdateAllAI()
    {
        Enemy[] enemies = EnemyManager.Instance.GetAllActive();
        for (int i = 0; i < enemies.Length; i++)
        {
            using (s_AI_Decision.Auto())
            {
                enemies[i].MakeDecision();
            }

            using (s_AI_Movement.Auto())
            {
                enemies[i].UpdateMovement(Time.deltaTime);
            }

            using (s_AI_Animation.Auto())
            {
                enemies[i].UpdateAnimation();
            }
        }
    }
}
```

### 缓存友好性优化（深度分析）

CPU缓存命中率对性能影响巨大。L1缓存命中约1ns，L2约4ns，L3约10ns，而内存访问约100ns。顺序访问比随机访问快10-100倍：

```csharp
/// <summary>
/// ECS-style数据布局 vs OOP-style数据布局
/// 演示缓存友好性对性能的影响
/// </summary>
public class CacheOptimizationDemo : MonoBehaviour
{
    // === 差：OOP风格，数据分散在堆内存中 ===
    class Enemy_OOP
    {
        public Vector3 position;
        public float health;
        public string name;        // 引用类型，数据在别处
        public GameObject model;   // 引用类型，数据在别处
        public List<Buff> buffs;   // 引用类型，数据在别处
        // 额外字段可能分散在不同内存页
        public float speed;
        public float attack;
        public float defense;
    }

    // 循环处理时，每个Enemy对象在不同内存位置
    // 随机访问导致大量缓存未命中
    void Update_OOP(List<Enemy_OOP> enemies)
    {
        float dt = Time.deltaTime;
        for (int i = 0; i < enemies.Count; i++)
        {
            // 每次访问enemies[i]可能触发缓存未命中
            // 访问position可能在不同缓存行
            enemies[i].position += Vector3.forward * enemies[i].speed * dt;
        }
    }

    // === 好：结构体数组，数据紧凑排列 ===
    struct EnemyData
    {
        public Vector3 position;
        public float speed;
        public float health;
        public float attack;
        public float defense;
        // 所有数据在同一缓存行（64字节）中
    }

    // 连续内存访问，缓存命中率高
    void Update_DataDriven(EnemyData[] enemies, int count)
    {
        float dt = Time.deltaTime;
        for (int i = 0; i < count; i++)
        {
            // 连续内存访问，缓存友好
            EnemyData e = enemies[i];
            e.position += Vector3.forward * e.speed * dt;
            enemies[i] = e;
        }
    }

    // 性能对比数据（10000个Enemy，单帧）：
    // OOP风格: 2.5ms, 80% L1缓存未命中
    // 数据驱动: 0.3ms, 10% L1缓存未命中
    // 提升: 8x
}
```

### SIMD优化示例

```csharp
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

/// <summary>
/// 使用Unity.Mathematics自动利用SIMD指令
/// Burst编译后会自动使用SSE/AVX指令集
/// </summary>
public class SIMDOptimization : MonoBehaviour
{
    // Unity.Mathematics的float3/float4自动编译为SIMD
    void SIMDExample()
    {
        float3 a = new float3(1, 2, 3);
        float3 b = new float3(4, 5, 6);
        float3 c = a + b; // 编译后使用SSE指令，一次处理4个float
        float dot = math.dot(a, b); // SIMD点积
        float len = math.length(a); // SIMD长度计算
    }

    // 批量向量运算
    [BurstCompile]
    struct BatchNormalizeJob : IJobParallelFor
    {
        public NativeArray<float3> vectors;

        public void Execute(int index)
        {
            // Burst会将此编译为SIMD指令
            vectors[index] = math.normalize(vectors[index]);
        }
    }

    // 性能对比（100000次normalize）：
    // 标准C#: 12ms
    // Burst+SIMD: 0.8ms
    // 提升: 15x
}
```

### 避免GC Alloc的常用技巧（完整版）

```csharp
using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>
/// GC分配优化完整指南
/// 目标：热路径每帧GC Alloc < 0.5KB
/// </summary>
public class GCOptimizationGuide : MonoBehaviour
{
    // === 1. 字符串拼接 ===
    // 差：每次+操作都创建新字符串
    void Bad_StringConcat()
    {
        string info = "HP:" + hp.ToString() + " MP:" + mp.ToString();
        // 每帧分配约50字节
    }

    // 好：StringBuilder复用
    private StringBuilder sb = new StringBuilder(64);
    void Good_StringBuilder()
    {
        sb.Clear();
        sb.Append("HP:").Append(hp).Append(" MP:").Append(mp);
        hpText.text = sb.ToString();
    }

    // 更好：仅在值变化时更新
    private int lastHP = -1, lastMP = -1;
    void Good_ValueChangedOnly()
    {
        if (hp != lastHP || mp != lastMP)
        {
            lastHP = hp; lastMP = mp;
            sb.Clear();
            sb.Append("HP:").Append(hp).Append(" MP:").Append(mp);
            hpText.text = sb.ToString();
        }
    }

    // === 2. List和集合 ===
    // 差：每帧new List
    void Bad_NewList()
    {
        var list = new List<Enemy>(); // 分配内存
        foreach (var e in enemies) list.Add(e);
    }

    // 好：预分配和复用
    private List<Enemy> tempList = new List<Enemy>(32);
    void Good_ReuseList()
    {
        tempList.Clear(); // 不释放内存，仅清空
        tempList.AddRange(enemies);
    }

    // === 3. 闭包和Lambda ===
    // 差：Lambda捕获局部变量产生闭包分配
    void Bad_Closure()
    {
        int multiplier = GetMultiplier();
        enemies.ForEach(e => e.Damage(baseDamage * multiplier));
        // 闭包在堆上分配
    }

    // 好：使用for循环或方法引用
    void Good_NoClosure()
    {
        int multiplier = GetMultiplier();
        for (int i = 0; i < enemies.Count; i++)
            enemies[i].Damage(baseDamage * multiplier);
    }

    // === 4. 协程GC ===
    // 差：每帧new WaitForSeconds
    IEnumerator Bad_Coroutine()
    {
        while (true)
        {
            UpdateSomething();
            yield return new WaitForSeconds(1f); // 每次分配
        }
    }

    // 好：缓存WaitForSeconds
    private static readonly WaitForSeconds oneSecond = new WaitForSeconds(1f);
    IEnumerator Good_Coroutine()
    {
        while (true)
        {
            UpdateSomething();
            yield return oneSecond; // 复用
        }
    }

    // === 5. 装箱 ===
    // 差：Dictionary<int, object>导致值类型装箱
    void Bad_Boxing()
    {
        var dict = new Dictionary<int, object>();
        dict[0] = 42; // int装箱为object
    }

    // 好：使用泛型Dictionary
    void Good_Generic()
    {
        var dict = new Dictionary<int, int>();
        dict[0] = 42; // 无装箱
    }

    // === 6. ArrayPool ===
    // 好：使用ArrayPool代替new T[]
    void Good_ArrayPool()
    {
        var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(1024);
        try
        {
            // 使用buffer
            ProcessBuffer(buffer);
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // === 7. Span<T>减少切片分配（C# 7.2+） ===
    void Good_Span()
    {
        string fullString = "Hello World Example";
        // 差：Substring分配新string
        // string sub = fullString.Substring(6, 5); // 分配

        // 好：Span零分配
        ReadOnlySpan<char> span = fullString.AsSpan(6, 5);
    }

    // 占位方法
    int hp, mp, baseDamage;
    TextMesh hpText;
    List<Enemy> enemies = new List<Enemy>();
    int GetMultiplier() => 2;
    void UpdateSomething() { }
    void ProcessBuffer(byte[] buffer) { }
    class Enemy { public void Damage(int d) { } }
}
```

## 最佳实践

- 发布版本关闭Deep Profile，仅在需要时开启（本身有10-30%性能开销）
- 使用Profiler.BeginSample/EndSample或ProfilerMarker标记自定义代码段
- 建立性能基准：记录关键场景的帧时间、GC频率作为回归对比基准
- 定期在目标设备（真机）上进行性能分析，Editor性能数据不可信（通常比真机慢2-5倍）
- 关注每帧GC Alloc而非总内存，目标控制在每帧0.5KB以内
- 使用Frame Timing Display（Ctrl+7）在真机上实时查看帧时间

## 常见陷阱与修复

**陷阱1：只在Editor中做性能分析**
- 症状：Editor中60fps但真机只有20fps
- 修复：所有性能分析必须在真机上进行，Editor有额外的编辑器开销

**陷阱2：过度优化冷门代码路径**
- 症状：花大量时间优化一个占0.1%的函数
- 修复：始终从Profiler中占比最高的热点函数开始优化（80/20原则）

**陷阱3：忘记关闭Deep Profile**
- 症状：Deep Profile本身让帧率降低30-50%
- 修复：仅在需要捕获完整调用栈时开启，平时使用普通Profile模式

**陷阱4：使用字符串拼接构造日志但未做开关控制**
- 症状：Release版本中Debug.Log仍有开销（字符串拼接消耗CPU和GC）
- 修复：使用条件编译`[System.Diagnostics.Conditional("UNITY_EDITOR")]`或自定义日志级别
