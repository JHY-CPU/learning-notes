# 更新方法（Update Method）

## 核心概念

更新方法模式让每个游戏对象在每一帧有机会执行自己的逻辑更新。这是最基础的游戏行为模式，几乎所有游戏对象都通过 Update 方法驱动行为。它是游戏引擎与普通软件最本质的区别之一。

### 为什么需要 Update 方法？

```
普通软件：事件驱动
用户点击按钮 → 执行回调 → 等待下一个事件
大部分时间 CPU 空闲

游戏：持续驱动
while (gameRunning) {
    for (obj in allObjects)
        obj.Update(dt);   // 每个对象每帧都有机会"做事"
    Render();
}
即使没有输入，敌人在巡逻、粒子在飞散、天气在变化
```

### Tick 驱动逻辑

```csharp
// ========== 游戏对象基类 ==========
public abstract class GameObject
{
    public bool IsActive = true;
    public bool IsDestroyed = false;
    public int UpdateOrder = 0; // 更新优先级

    // 生命周期方法
    public virtual void Awake() { }           // 创建后立即调用
    public virtual void Start() { }           // 第一帧Update前调用
    public abstract void Update(float dt);    // 每帧调用
    public virtual void LateUpdate(float dt) { } // 所有Update后调用
    public virtual void FixedUpdate(float dt) { } // 固定步长更新
    public virtual void OnDestroy() { }       // 销毁前调用

    // 状态标记
    internal bool _started = false;
    internal bool _awakened = false;
}

// ========== 具体对象实现 ==========

public class Enemy : GameObject
{
    public Vector3 Position;
    public float Speed = 3f;
    public Vector3 TargetPosition;
    public float DetectionRange = 10f;
    public EnemyState State = EnemyState.Patrol;

    private int currentPatrolIndex = 0;
    private List<Vector3> patrolPoints;
    private float stateTimer = 0f;

    public override void Start()
    {
        patrolPoints = GetPatrolRoute();
    }

    public override void Update(float dt)
    {
        stateTimer += dt;

        switch (State)
        {
            case EnemyState.Patrol:
                UpdatePatrol(dt);
                break;
            case EnemyState.Chase:
                UpdateChase(dt);
                break;
            case EnemyState.Attack:
                UpdateAttack(dt);
                break;
            case EnemyState.Idle:
                UpdateIdle(dt);
                break;
        }
    }

    private void UpdatePatrol(float dt)
    {
        if (patrolPoints == null || patrolPoints.Count == 0) return;

        Vector3 target = patrolPoints[currentPatrolIndex];
        MoveToward(target, Speed * 0.5f, dt);

        if (Vector3.Distance(Position, target) < 0.5f)
        {
            currentPatrolIndex = (currentPatrolIndex + 1) % patrolPoints.Count;
            State = EnemyState.Idle;
            stateTimer = 0f;
        }

        // 检测玩家
        if (PlayerInRange())
            State = EnemyState.Chase;
    }

    private void UpdateChase(float dt)
    {
        Vector3 playerPos = GameManager.Instance.Player.Position;
        MoveToward(playerPos, Speed, dt);

        if (Vector3.Distance(Position, playerPos) < 2f)
            State = EnemyState.Attack;
        if (!PlayerInRange())
            State = EnemyState.Patrol;
    }

    private void UpdateAttack(float dt)
    {
        if (stateTimer >= 1f) // 攻击间隔
        {
            PerformAttack();
            stateTimer = 0f;
        }

        Vector3 playerPos = GameManager.Instance.Player.Position;
        if (Vector3.Distance(Position, playerPos) > 2f)
            State = EnemyState.Chase;
    }

    private void UpdateIdle(float dt)
    {
        if (stateTimer >= 2f) // 等待2秒后继续巡逻
        {
            State = EnemyState.Patrol;
            stateTimer = 0f;
        }
        if (PlayerInRange())
            State = EnemyState.Chase;
    }

    private void MoveToward(Vector3 target, float speed, float dt)
    {
        Vector3 dir = (target - Position).normalized;
        Position += dir * speed * dt;
    }

    private bool PlayerInRange() =>
        Vector3.Distance(Position, GameManager.Instance.Player.Position) < DetectionRange;

    private void PerformAttack() { /* ... */ }
    private List<Vector3> GetPatrolRoute() { return new List<Vector3>(); }
}

public enum EnemyState { Idle, Patrol, Chase, Attack }

public class Bullet : GameObject
{
    public Vector3 Position;
    public Vector3 Velocity;
    public float Lifetime = 3f;
    public float Damage = 25f;

    public override void Update(float dt)
    {
        Position += Velocity * dt;
        Lifetime -= dt;

        if (Lifetime <= 0)
        {
            Destroy(this);
            return;
        }

        // 碰撞检测
        var hit = Physics.Raycast(Position, Velocity.normalized, Velocity.magnitude * dt);
        if (hit != null)
        {
            hit.Object.GetComponent<HealthComponent>()?.TakeDamage(Damage);
            VFXManager.Instance.SpawnHitEffect(Position);
            Destroy(this);
        }
    }
}

public class ParticleEffect : GameObject
{
    public Vector3[] Positions;
    public Vector3[] Velocities;
    public float[] Lifetimes;
    public Color[] Colors;
    public int Count;

    // 批量更新：一次遍历处理所有粒子，缓存友好
    public override void Update(float dt)
    {
        for (int i = 0; i < Count; i++)
        {
            Positions[i] += Velocities[i] * dt;
            Velocities[i] += Vector3.down * 9.8f * dt; // 重力
            Lifetimes[i] -= dt;

            // 渐隐
            float alpha = Mathf.Clamp01(Lifetimes[i] / 2f);
            Colors[i] = new Color(Colors[i].r, Colors[i].g, Colors[i].b, alpha);

            if (Lifetimes[i] <= 0)
            {
                // 交换到最后并减少计数
                Swap(i, Count - 1);
                Count--;
                i--;
            }
        }
    }

    private void Swap(int a, int b) { /* ... */ }
}
```

### 场景管理器与优先级排序

```csharp
/// <summary>
/// 场景管理器——驱动所有游戏对象的更新
/// </summary>
public class Scene
{
    private List<GameObject> objects = new();
    private List<GameObject> pendingAdd = new();
    private List<GameObject> pendingDestroy = new();
    private bool needsSort = true;
    private bool isUpdating = false;

    public int ObjectCount => objects.Count;

    /// <summary>
    /// 添加对象到场景
    /// </summary>
    public void Add(GameObject obj)
    {
        if (isUpdating)
        {
            pendingAdd.Add(obj); // 正在更新中，延迟添加
        }
        else
        {
            objects.Add(obj);
            needsSort = true;
        }
    }

    /// <summary>
    /// 标记销毁对象（不立即移除，避免遍历时修改集合）
    /// </summary>
    public void Destroy(GameObject obj)
    {
        obj.IsDestroyed = true;
        if (isUpdating)
            pendingDestroy.Add(obj);
        else
            RemoveImmediately(obj);
    }

    /// <summary>
    /// 主更新循环——每帧调用
    /// </summary>
    public void UpdateAll(float deltaTime)
    {
        isUpdating = true;

        // 1. 按优先级排序
        if (needsSort)
        {
            objects.Sort((a, b) => a.UpdateOrder.CompareTo(b.UpdateOrder));
            needsSort = false;
        }

        // 2. 首帧初始化
        for (int i = 0; i < objects.Count; i++)
        {
            var obj = objects[i];
            if (!obj.IsActive || obj.IsDestroyed) continue;

            if (!obj._awakened)
            {
                obj.Awake();
                obj._awakened = true;
            }
            if (!obj._started)
            {
                obj.Start();
                obj._started = true;
            }
        }

        // 3. FixedUpdate（固定步长）
        // （由外部累加器控制调用频率）

        // 4. Update（每帧调用）
        for (int i = 0; i < objects.Count; i++)
        {
            var obj = objects[i];
            if (obj.IsActive && !obj.IsDestroyed)
                obj.Update(deltaTime);
        }

        // 5. LateUpdate（所有Update之后）
        for (int i = 0; i < objects.Count; i++)
        {
            var obj = objects[i];
            if (obj.IsActive && !obj.IsDestroyed)
                obj.LateUpdate(deltaTime);
        }

        isUpdating = false;

        // 6. 处理延迟添加/销毁
        FlushPending();
    }

    /// <summary>
    /// FixedUpdate——固定步长更新
    /// </summary>
    public void FixedUpdateAll(float fixedDeltaTime)
    {
        foreach (var obj in objects)
        {
            if (obj.IsActive && !obj.IsDestroyed)
                obj.FixedUpdate(fixedDeltaTime);
        }
    }

    private void FlushPending()
    {
        foreach (var obj in pendingAdd)
        {
            objects.Add(obj);
            needsSort = true;
        }
        pendingAdd.Clear();

        foreach (var obj in pendingDestroy)
            RemoveImmediately(obj);
        pendingDestroy.Clear();
    }

    private void RemoveImmediately(GameObject obj)
    {
        obj.OnDestroy();
        objects.Remove(obj);
    }

    /// <summary>
    /// 查找满足条件的对象
    /// </summary>
    public T FindObjectOfType<T>() where T : GameObject
    {
        return objects.OfType<T>().FirstOrDefault();
    }

    public List<T> FindObjectsOfType<T>() where T : GameObject
    {
        return objects.OfType<T>().ToList();
    }
}
```

### 优先级排序详解

```csharp
// 更新优先级定义
// 数值越小越先执行
public static class UpdatePriority
{
    public const int InputSystem = 0;      // 先读取输入
    public const int Camera = 5;            // 相机更新
    public const int Physics = 10;          // 物理模拟
    public const int AI = 20;              // AI决策
    public const int Gameplay = 30;        // 游戏逻辑
    public const int Animation = 40;       // 动画更新
    public const int Movement = 50;        // 应用位移
    public const int Particles = 60;       // 粒子效果
    public const int UI = 70;             // UI更新
    public const int Audio = 80;          // 音频更新
    public const int Cleanup = 100;       // 清理标记删除的对象
}

// 使用
public class InputHandler : GameObject
{
    public override int UpdateOrder => UpdatePriority.InputSystem;
    public override void Update(float dt) { /* 处理输入 */ }
}

public class EnemyAI : GameObject
{
    public override int UpdateOrder => UpdatePriority.AI;
    public override void Update(float dt) { /* AI决策 */ }
}

public class CharacterMovement : GameObject
{
    public override int UpdateOrder => UpdatePriority.Movement;
    public override void Update(float dt) { /* 应用位移 */ }
}
```

### 批量更新优化

```csharp
// 差：逐个Update，10万次虚函数调用，缓存不友好
foreach (var particle in particles)
    particle.Update(dt);

// 好：批量更新，一次遍历处理所有数据，内存连续
public class ParticleSystem
{
    // 数据紧凑排列在数组中，缓存友好
    private Vector3[] positions;  // SOA (Structure of Arrays)
    private Vector3[] velocities;
    private float[] lifetimes;
    private Color[] colors;
    private int count;

    public void BatchUpdate(float dt)
    {
        for (int i = 0; i < count; i++)
        {
            positions[i] += velocities[i] * dt;
            velocities[i].y -= 9.8f * dt;
            lifetimes[i] -= dt;

            if (lifetimes[i] <= 0)
            {
                // 交换移除（O(1)）
                positions[i] = positions[count - 1];
                velocities[i] = velocities[count - 1];
                lifetimes[i] = lifetimes[count - 1];
                colors[i] = colors[count - 1];
                count--;
                i--;
            }
        }
    }

    // 性能对比（10万粒子）：
    // 逐个Update：~8ms/帧
    // 批量Update：~1.2ms/帧（快6-7倍）
}
```

### 事件驱动替代轮询

```csharp
// 差：每帧轮询检查条件
public class QuestChecker : GameObject
{
    public override void Update(float dt)
    {
        // 每帧检查100个任务条件，浪费CPU
        foreach (var quest in allQuests)
        {
            if (quest.Condition.IsMet())
                quest.Complete();
        }
    }
}

// 好：事件驱动，条件满足时才触发
public class QuestSystem
{
    public void Initialize()
    {
        EventBus.Subscribe<DamageEvent>(CheckDamageQuests);
        EventBus.Subscribe<ItemPickupEvent>(CheckItemQuests);
        EventBus.Subscribe<KillEvent>(CheckKillQuests);
        // 只在相关事件发生时才检查
    }

    void CheckDamageQuests(DamageEvent evt)
    {
        var damageQuests = activeQuests.Where(q => q.Type == QuestType.DealDamage);
        foreach (var quest in damageQuests)
        {
            quest.Progress += (int)evt.Amount;
            if (quest.IsComplete)
                quest.Complete();
        }
    }
}
```

### Unity 的 Update 生命周期

```csharp
public class MyComponent : MonoBehaviour
{
    // Unity 引擎每帧调用，可变步长
    void Update()
    {
        float dt = Time.deltaTime;
        transform.Rotate(Vector3.up, 90f * dt);
    }

    // 固定步长（默认50Hz = 0.02s）
    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        rb.AddForce(Vector3.forward * 10f); // 物理操作
    }

    // 所有Update执行完后调用
    void LateUpdate()
    {
        // 相机跟随：确保在角色移动后再更新相机位置
        cameraTransform.position = target.position + offset;
    }

    // Unity 的完整生命周期顺序：
    // Awake → OnEnable → Start → FixedUpdate → Update → LateUpdate → OnDisable → OnDestroy
}
```

### Unreal Engine 的 Tick 系统

```cpp
// UE 的 Tick 提供了更精细的控制
UCLASS()
class AMyActor : public AActor
{
    GENERATED_BODY()

public:
    AMyActor()
    {
        // 配置 Tick 属性
        PrimaryActorTick.bCanEverTick = true;
        PrimaryActorTick.TickGroup = TG_PrePhysics; // 更新阶段
        PrimaryActorTick.TickInterval = 0.0f; // 每帧都Tick，设为0.1则每0.1秒Tick一次
    }

    virtual void Tick(float DeltaTime) override
    {
        Super::Tick(DeltaTime);

        // 自定义逻辑
        FVector NewLocation = GetActorLocation();
        NewLocation.Z += FMath::Sin(RunningTime) * 50.f;
        SetActorLocation(NewLocation);
        RunningTime += DeltaTime;
    }

    // TickGroup 选项：
    // TG_PrePhysics   —— 物理模拟前（输入处理）
    // TG_DuringPhysics —— 物理模拟中
    // TG_PostPhysics   —— 物理模拟后（角色移动结果）
    // TG_PostUpdateWork —— 渲染前最后更新

private:
    float RunningTime = 0.f;
};
```

## 方案对比

| 方案 | 实现难度 | 性能 | 缓存友好 | 并行化 | 适用场景 |
|------|---------|------|---------|--------|---------|
| 逐个 Update | 低 | 差 | 差 | 不支持 | 小型项目 |
| 优先级排序 | 中 | 中 | 差 | 不支持 | 中型项目 |
| 批量更新 | 中 | 优秀 | 优秀 | 支持 | 粒子、子弹等大量相似对象 |
| 事件驱动 | 中 | 优秀 | N/A | 支持 | 条件触发逻辑 |
| ECS System | 高 | 极好 | 极好 | 原生支持 | 大型项目 |

## 常见陷阱与解决方案

1. **Update 开销过大**：大量 MonoBehaviour 都有 Update 导致每帧函数调用过多。解决方案：合并 Update、使用事件驱动、减少活跃对象
2. **更新顺序依赖**：A对象的Update依赖B对象的Update结果。解决方案：使用优先级排序，或分离到不同生命周期方法（Update vs LateUpdate）
3. **dt 使用错误**：在 FixedUpdate 中使用 Time.deltaTime 而非 Time.fixedDeltaTime。解决方案：明确区分两种 dt
4. **对象销毁后 Update 仍在执行**：Destroy 后本帧内对象仍然存活。解决方案：使用 IsDestroyed 标记检查
5. **性能热点**：某个对象的 Update 特别耗时。解决方案：使用 Profiler 定位，考虑批量更新或Job System

## Unity 实现

```csharp
// 性能优化：禁用不需要的 Update
public class OptimizedComponent : MonoBehaviour
{
    void OnEnable()
    {
        // 只在需要时才启用 Update
        if (!needsUpdate)
            enabled = false;
    }

    // 使用 InvokeRepeating 替代 Update 中的计时器
    void Start()
    {
        InvokeRepeating(nameof(Tick), 0f, 0.1f); // 每0.1秒调用一次
    }

    void Tick()
    {
        // 低频更新逻辑
    }
}
```

## 实际使用案例

- **Unity 的 MonoBehaviour** 提供 Update/FixedUpdate/LateUpdate 三个生命周期方法
- **Unreal Engine 的 Tick 函数** 提供更精细的控制（TickGroup、TickInterval）
- **《魔兽争霸3》** 的触发器系统本质是基于 Update 的事件检查
- **性能敏感的项目** 会减少 Update 使用，转用 ECS 批量更新或事件驱动替代轮询
- **《守望先锋》** 使用 60Hz 固定 tick 率驱动所有游戏逻辑更新
