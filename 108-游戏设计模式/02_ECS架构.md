# ECS架构

## 核心概念

ECS（Entity-Component-System）是一种数据驱动的游戏架构模式，将数据（Component）与逻辑（System）完全分离，以 Entity 作为索引组织数据。ECS 的核心理念是"面向数据编程"（Data-Oriented Design），通过优化数据在内存中的布局来最大化 CPU 缓存命中率。

### 为什么需要 ECS？OOP 的困境

传统的面向对象继承体系在游戏开发中会遇到"组合爆炸"问题：

```
OOP 继承体系：
GameObject
  └─ Character
       ├─ Player
       │    ├─ Warrior     (可移动 + 可战斗 + 可近战)
       │    ├─ Mage        (可移动 + 可战斗 + 可施法)
       │    └─ Archer      (可移动 + 可战斗 + 可远程)
       └─ NPC
            ├─ Shopkeeper  (可对话 + 可交易)
            └─ Monster     (可移动 + 可战斗 + 可AI)

问题1：如果要一个"会飞的会游泳的会施法的商人"怎么办？
问题2：Warrior和Monster都有"可战斗"逻辑，需要在两个类中重复实现
问题3：增加一个新能力（如"可骑乘"）需要修改大量类
```

```
ECS 方式：
Entity 1 = {Position, Health, MeleeAttack, PlayerInput}
Entity 2 = {Position, Health, MagicAttack, PlayerInput}
Entity 3 = {Position, Health, FlyingAbility, SwimmingAbility, MagicAttack, AIBehavior}
Entity 4 = {Position, DialogueTree, TradeInventory}

System 只关心"哪些实体拥有我需要的组件组合"
```

### 三大要素详解

```
Entity（实体）：
    仅仅是一个唯一ID（通常是32位或64位整数），不包含任何数据或逻辑。
    它就像数据库中的一行主键，用来索引对应的组件数据。
    Entity 可以被创建和销毁，销毁时自动移除其所有组件。

Component（组件）：
    纯数据结构（struct），没有方法，描述实体的某一类属性。
    Position 组件只存储坐标(x, y, z)。
    Health 组件只存储当前生命值和最大生命值。
    不应该有 Update() 方法——逻辑全部在 System 中。

System（系统）：
    纯逻辑，遍历拥有特定组件组合的实体，对数据执行操作。
    MovementSystem 查询所有同时拥有 Position 和 Velocity 的实体。
    DamageSystem 查询所有同时拥有 Health 和 DamageEvent 的实体。
    System 不持有数据，只持有逻辑。
```

### Archetype 稀疏集 vs 稠密集架构

ECS 内部数据存储有多种实现方式，性能特征差异显著：

**Archetype（原型）方式**：将拥有相同组件组合的实体数据紧凑存储在一起。

```
Archetype 存储示意：
Archetype [Position, Velocity, Health]:
    EntityID:  [1,  3,  7,  12, ...]
    Position:  [p1, p3, p7, p12, ...]  ← 连续内存块
    Velocity:  [v1, v3, v7, v12, ...]  ← 连续内存块
    Health:    [h1, h3, h7, h12, ...]  ← 连续内存块

MovementSystem 遍历时：
    一次性读取整块 Position 数组和 Velocity 数组
    CPU 预取器能有效工作，缓存命中率极高
    缺点：添加/移除组件需要移动数据到新 Archetype
```

**稀疏集（Sparse Set）方式**：每个组件类型独立存储，用稀疏数组索引。

```
Sparse Set 存储示意：
Position 稠密数组: [p1, p3, p7, p12, ...]
Position 稀疏索引: EntityID → 稠密数组下标

Velocity 稠密数组: [v1, v3, v12, ...]  ← 注意：实体7没有Velocity
Velocity 稀疏索引: EntityID → 稠密数组下标

优点：动态添加/移除组件非常快，只需操作对应数组
缺点：不同组件的同一实体在内存中不连续，遍历时缓存命中率稍低
```

### C# ECS 完整实现（轻量级）

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

// ========== Entity 定义 ==========
public struct Entity : IEquatable<Entity>
{
    public readonly uint Id;
    public readonly ushort Version; // 用于检测"过期引用"

    public Entity(uint id, ushort version)
    {
        Id = id;
        Version = version;
    }

    public bool Equals(Entity other) => Id == other.Id && Version == other.Version;
    public override int GetHashCode() => (int)Id;
    public override string ToString() => $"Entity({Id}v{Version})";
}

// ========== World：ECS 的核心容器 ==========
public class World
{
    private uint nextEntityId = 0;
    private Dictionary<uint, ushort> entityVersions = new();
    private HashSet<uint> aliveEntities = new();

    // 每种组件类型一个字典：EntityID → 组件数据
    private Dictionary<Type, Dictionary<uint, object>> componentStores = new();

    // 所有注册的 System
    private List<ISystem> systems = new();

    public Entity CreateEntity()
    {
        uint id = nextEntityId++;
        ushort version = 1;
        entityVersions[id] = version;
        aliveEntities.Add(id);
        return new Entity(id, version);
    }

    public void DestroyEntity(Entity entity)
    {
        if (!aliveEntities.Contains(entity.Id)) return;

        aliveEntities.Remove(entity.Id);
        entityVersions[entity.Id]++; // 递增版本号

        // 移除所有组件
        foreach (var store in componentStores.Values)
            store.Remove(entity.Id);
    }

    // 添加组件
    public void AddComponent<T>(Entity entity, T component) where T : struct
    {
        var type = typeof(T);
        if (!componentStores.ContainsKey(type))
            componentStores[type] = new Dictionary<uint, object>();

        componentStores[type][entity.Id] = component;
    }

    // 获取组件引用
    public ref T GetComponent<T>(Entity entity) where T : struct
    {
        return ref ((Dictionary<uint, T>)GetOrCreateTypedStore<T>())[entity.Id];
    }

    // 检查实体是否拥有某组件
    public bool HasComponent<T>(Entity entity) where T : struct
    {
        var type = typeof(T);
        return componentStores.ContainsKey(type) &&
               componentStores[type].ContainsKey(entity.Id);
    }

    // 查询拥有所有指定组件的实体
    public List<Entity> Query<T1, T2>()
        where T1 : struct
        where T2 : struct
    {
        var result = new List<Entity>();
        var store1 = GetOrCreateTypedStore<T1>();
        var store2 = GetOrCreateTypedStore<T2>();

        foreach (var entityId in store1.Keys)
        {
            if (store2.ContainsKey(entityId) && aliveEntities.Contains(entityId))
                result.Add(new Entity(entityId, entityVersions[entityId]));
        }
        return result;
    }

    public void RegisterSystem(ISystem system) => systems.Add(system);

    public void UpdateSystems(float deltaTime)
    {
        foreach (var system in systems)
            system.Update(this, deltaTime);
    }

    private Dictionary<uint, T> GetOrCreateTypedStore<T>() where T : struct
    {
        var type = typeof(T);
        if (!componentStores.ContainsKey(type))
            componentStores[type] = new Dictionary<uint, object>();

        // 如果存储类型不匹配，转换之
        if (componentStores[type] is not Dictionary<uint, T> typed)
        {
            typed = new Dictionary<uint, T>();
            foreach (var kvp in componentStores[type])
                typed[kvp.Key] = (T)kvp.Value;
            componentStores[type] = typed;
        }
        return typed;
    }
}

// ========== Component 定义（纯数据） ==========
public struct Position { public float X, Y, Z; }
public struct Velocity { public float X, Y, Z; }
public struct Health { public float Current; public float Max; }
public struct DamageEvent { public uint TargetId; public float Amount; }

// ========== System 定义（纯逻辑） ==========
public interface ISystem
{
    void Update(World world, float deltaTime);
}

public class MovementSystem : ISystem
{
    public void Update(World world, float deltaTime)
    {
        foreach (var entity in world.Query<Position, Velocity>())
        {
            ref var pos = ref world.GetComponent<Position>(entity);
            ref var vel = ref world.GetComponent<Velocity>(entity);

            pos.X += vel.X * deltaTime;
            pos.Y += vel.Y * deltaTime;
            pos.Z += vel.Z * deltaTime;
        }
    }
}

public class DamageSystem : ISystem
{
    public void Update(World world, float deltaTime)
    {
        // 处理伤害事件
        // （实际项目中会用事件队列，这里简化）
    }
}

public class HealthRegenSystem : ISystem
{
    public void Update(World world, float deltaTime)
    {
        foreach (var entity in world.Query<Position, Health>())
        {
            ref var health = ref world.GetComponent<Health>(entity);
            if (health.Current < health.Max)
            {
                health.Current = Math.Min(health.Max, health.Current + 5f * deltaTime);
            }
        }
    }
}
```

### Unity DOTS（Data-Oriented Technology Stack）

Unity 的 DOT S 是 ECS 架构的工业级实现，结合 Burst 编译器和 C# Job System。

```csharp
using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;
using Unity.Burst;

// ===== Component 定义 =====
public struct Speed : IComponentData
{
    public float Value;
}

public struct HealthData : IComponentData
{
    public float Current;
    public float Max;
}

public struct EnemyTag : IComponentData { }

public struct Spawner : IComponentData
{
    public Entity Prefab;
    public float SpawnInterval;
    public float NextSpawnTime;
}

// ===== System 定义 =====
// [BurstCompile] 可以将C#代码编译为高度优化的native代码
[BurstCompile]
public partial struct MovementSystem : ISystem
{
    // [BurstCompile] 标记让Burst编译器优化此方法
    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        float dt = SystemAPI.Time.DeltaTime;

        // SystemAPI.Query 自动遍历所有匹配的实体
        // RefRW = 读写引用, RefRO = 只读引用
        foreach (var (transform, speed) in SystemAPI.Query<
            RefRW<LocalTransform>,
            RefRO<Speed>>())
        {
            // 直接操作连续内存中的数据，缓存友好
            transform.ValueRW.Position += math.forward() * speed.ValueRO.Value * dt;
        }
    }
}

[BurstCompile]
public partial struct SpawnSystem : ISystem
{
    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        // EntityCommandBuffer 用于在System中安全地创建/销毁实体
        var ecb = new EntityCommandBuffer(Unity.Collections.Allocator.Temp);
        float time = (float)SystemAPI.Time.ElapsedTime;

        foreach (var spawner in SystemAPI.Query<RefRW<Spawner>>())
        {
            if (time >= spawner.ValueRO.NextSpawnTime)
            {
                var entity = ecb.Instantiate(spawner.ValueRO.Prefab);
                ecb.SetComponent(entity, new LocalTransform
                {
                    Position = float3.zero,
                    Rotation = quaternion.identity,
                    Scale = 1f
                });
                spawner.ValueRW.NextSpawnTime = time + spawner.ValueRO.SpawnInterval;
            }
        }

        ecb.Playback(state.EntityManager);
        ecb.Dispose();
    }
}

// 使用 IJobEntity 进行并行化
[BurstCompile]
public partial struct ParallelMovementJob : IJobEntity
{
    public float DeltaTime;

    // 方法参数名必须匹配组件字段名，或使用 [EntityIndexInQuery] 属性
    public void Execute(ref LocalTransform transform, in Speed speed)
    {
        transform.Position += math.forward() * speed.Value * DeltaTime;
    }
}
```

### 缓存友好的迭代

Archetype ECS 的核心性能优势在于数据局部性：

```
传统 OOP 遍历（缓存不友好）：
Enemy[0] 在内存地址 0x1000 (Position, Health, AI, Render, Physics...)
Enemy[1] 在内存地址 0x2000 (Position, Health, AI, Render, Physics...)
遍历所有敌人更新位置时，每次只用到 Position 字段，
但 CPU 缓存行会把整个对象加载进来，大量数据是无用的。

Archetype ECS 遍历（缓存友好）：
Archetype [Position, Velocity]:
    Position数组: [0x1000] = [p0, p1, p2, p3, ..., pN]
    Velocity数组: [0x2000] = [v0, v1, v2, v3, ..., vN]
MovementSystem 遍历时，连续读取 Position 和 Velocity 数组，
CPU 预取器能高效预测下一步读取的内存，缓存命中率接近100%。

性能差距：在拥有10万实体的场景中，ECS遍历速度可达OOP的5-10倍。
```

## 方案对比

| 特性 | 传统 OOP | 组件模式 | ECS (Sparse Set) | ECS (Archetype) |
|------|---------|---------|-------------------|-----------------|
| 数据布局 | 对象内联 | 对象内联 | 每组件独立数组 | 同原型紧凑排列 |
| 缓存友好度 | 差 | 差 | 中等 | 极佳 |
| 添加/移除组件 | N/A | 快 | 快 | 慢（需移动数据） |
| 查询速度 | O(n)遍历 | O(n)遍历 | O(min)交集 | O(1)直接定位 |
| 并行友好度 | 差 | 差 | 好 | 极好 |
| 学习曲线 | 低 | 低 | 中 | 高 |
| 调试难度 | 低 | 中 | 高 | 很高 |

## 常见陷阱与解决方案

1. **过度设计**：小型游戏不需要完整的ECS。解决方案：项目规模小用组件模式即可
2. **调试困难**：数据分散在各个数组中。解决方案：使用ECS调试器（如Unity的Entity Debugger）
3. **系统执行顺序**：不同System之间可能有依赖。解决方案：显式定义System执行顺序/依赖
4. **组件间通信**：ECS中不能直接调用其他组件的方法。解决方案：使用事件队列或命令缓冲区
5. **内存碎片**：频繁创建/销毁实体导致Sparse Set碎片。解决方案：定期整理或使用对象池

## 实际使用案例

- **Unity DOTS** 用于《城市：天际线2》处理数十万市民的路径和行为计算
- **《守望先锋》** 的网络同步基于ECS，确保确定性回放和观战功能
- **Rust 的 Bevy 引擎** 和 **C++ 的 EnTT 库** 是社区流行的开源ECS实现
- **EA 的《FIFA》系列** 使用自研ECS处理球员动画和物理
- **《我的世界》** 的区块系统虽然不是经典ECS，但数据驱动的思路类似
