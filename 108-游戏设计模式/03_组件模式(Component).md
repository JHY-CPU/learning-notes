# 组件模式（Component）

## 核心概念

组件模式通过将功能拆分为独立的可插拔组件来构建游戏对象，取代深层继承体系。每个组件负责单一职责，游戏对象通过组合不同组件来实现丰富多样的行为。这是 Unity 的核心设计哲学。

### 继承 vs 组合：为什么组件模式胜出

```
继承的问题——组合爆炸：
class Character { }
class FlyingCharacter : Character { }
class SwimmingCharacter : Character { }
class FlyingSwimmingCharacter : Character { } // 又飞又游？
class FlyingSwimmingInvisibleCharacter : Character { } // 又飞又游还会隐身？

每增加一个能力就需要新的子类，类的数量呈指数增长。
如果能力有N种，可能的组合有 2^N 种。

组件模式——能力即组件：
class GameObject {
    List<Component> components;
}
一个"又飞又游又隐身"的角色 = GameObject + Flying + Swimming + Invisibility
新增能力只需添加组件，无需创建新类。
```

### 完整组件系统实现

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

// ========== 基础组件抽象 ==========
public abstract class Component
{
    public GameObject Owner { get; internal set; }
    public bool Enabled = true;

    // 生命周期方法
    public virtual void Awake() { }          // 组件被添加后立即调用
    public virtual void Start() { }          // 第一帧Update之前调用
    public virtual void Update(float dt) { } // 每帧调用
    public virtual void LateUpdate(float dt) { } // 所有Update之后调用
    public virtual void OnDestroy() { }      // 组件被移除前调用
    public virtual void OnEnable() { }       // 组件启用时调用
    public virtual void OnDisable() { }      // 组件禁用时调用

    // 获取同实体上的其他组件
    public T GetComponent<T>() where T : Component => Owner.GetComponent<T>();
    public T AddComponent<T>() where T : Component, new() => Owner.AddComponent<T>();
    public void RemoveComponent<T>() where T : Component => Owner.RemoveComponent<T>();
}

// ========== 游戏对象（组件容器） ==========
public class GameObject
{
    public string Name;
    public Transform Transform = new Transform();
    public bool Active = true;

    private List<Component> components = new();
    private List<Component> pendingAdd = new();
    private List<Component> pendingRemove = new();
    private bool isUpdating = false;

    // 获取组件
    public T GetComponent<T>() where T : Component
    {
        return components.OfType<T>().FirstOrDefault();
    }

    public List<T> GetComponents<T>() where T : Component
    {
        return components.OfType<T>().ToList();
    }

    // 添加组件（支持运行时动态添加）
    public T AddComponent<T>() where T : Component, new()
    {
        var comp = new T();
        comp.Owner = this;

        if (isUpdating)
        {
            pendingAdd.Add(comp); // 正在更新中，延迟添加
        }
        else
        {
            components.Add(comp);
            comp.Awake();
        }
        return comp;
    }

    // 移除组件（支持运行时动态移除）
    public void RemoveComponent<T>() where T : Component
    {
        var comp = GetComponent<T>();
        if (comp == null) return;

        if (isUpdating)
        {
            pendingRemove.Add(comp);
        }
        else
        {
            comp.OnDestroy();
            components.Remove(comp);
        }
    }

    // 场景管理器每帧调用
    public void InternalUpdate(float dt)
    {
        if (!Active) return;

        isUpdating = true;

        // 第一帧初始化
        foreach (var comp in components)
        {
            if (comp.Enabled && !comp._started)
            {
                comp.Start();
                comp._started = true;
            }
        }

        // 更新
        foreach (var comp in components)
        {
            if (comp.Enabled)
                comp.Update(dt);
        }

        isUpdating = false;

        // 处理延迟添加/移除
        FlushPending();
    }

    private void FlushPending()
    {
        foreach (var comp in pendingAdd)
        {
            components.Add(comp);
            comp.Awake();
        }
        pendingAdd.Clear();

        foreach (var comp in pendingRemove)
        {
            comp.OnDestroy();
            components.Remove(comp);
        }
        pendingRemove.Clear();
    }
}

public class Transform
{
    public Vector3 Position = Vector3.Zero;
    public Vector3 Rotation = Vector3.Zero;
    public Vector3 Scale = Vector3.One;
}
```

### 实际游戏组件实现

```csharp
// ========== 健康值组件 ==========
public class HealthComponent : Component
{
    public float Current = 100f;
    public float Max = 100f;
    public float RegenRate = 0f; // 每秒回复量
    public bool IsDead => Current <= 0;

    // 事件，允许其他组件/系统监听
    public event Action<float, float> OnHealthChanged; // (current, max)
    public event Action OnDeath;

    public override void Update(float dt)
    {
        if (RegenRate > 0 && Current < Max && !IsDead)
        {
            Current = Math.Min(Max, Current + RegenRate * dt);
            OnHealthChanged?.Invoke(Current, Max);
        }
    }

    public void TakeDamage(float amount)
    {
        if (IsDead) return;
        Current = Math.Max(0, Current - amount);
        OnHealthChanged?.Invoke(Current, Max);

        if (IsDead)
            OnDeath?.Invoke();
    }

    public void Heal(float amount)
    {
        if (IsDead) return;
        Current = Math.Min(Max, Current + amount);
        OnHealthChanged?.Invoke(Current, Max);
    }
}

// ========== 移动组件 ==========
public class MoveComponent : Component
{
    public float Speed = 5f;
    public float Gravity = -9.8f;
    private float verticalVelocity = 0f;

    public override void Update(float dt)
    {
        // 获取输入组件的方向
        var input = Owner.GetComponent<InputComponent>();
        if (input == null) return;

        Vector3 move = input.MoveDirection * Speed;

        // 重力
        if (!IsGrounded())
            verticalVelocity += Gravity * dt;
        else
            verticalVelocity = 0f;

        move.Y = verticalVelocity;
        Owner.Transform.Position += move * dt;
    }

    private bool IsGrounded()
    {
        // 射线检测地面
        return Owner.Transform.Position.Y <= 0.1f;
    }
}

// ========== 输入组件 ==========
public class InputComponent : Component
{
    public Vector3 MoveDirection { get; private set; }
    public bool JumpPressed { get; private set; }
    public bool AttackPressed { get; private set; }

    public override void Update(float dt)
    {
        // 读取键盘/手柄输入
        MoveDirection = ReadMovementInput();
        JumpPressed = ReadJumpInput();
        AttackPressed = ReadAttackInput();
    }

    private Vector3 ReadMovementInput() { /* ... */ return Vector3.Zero; }
    private bool ReadJumpInput() { /* ... */ return false; }
    private bool ReadAttackInput() { /* ... */ return false; }
}

// ========== 武器组件 ==========
public class WeaponComponent : Component
{
    public float Damage = 25f;
    public float AttackSpeed = 2f; // 每秒攻击次数
    public float Range = 2f;
    public float Cooldown = 0f;

    public override void Update(float dt)
    {
        Cooldown = Math.Max(0, Cooldown - dt);

        var input = Owner.GetComponent<InputComponent>();
        if (input != null && input.AttackPressed && Cooldown <= 0)
        {
            PerformAttack();
            Cooldown = 1f / AttackSpeed;
        }
    }

    private void PerformAttack()
    {
        // 检测范围内的敌人
        var targets = FindTargetsInRange();
        foreach (var target in targets)
        {
            var health = target.GetComponent<HealthComponent>();
            health?.TakeDamage(Damage);
        }
    }

    private List<GameObject> FindTargetsInRange() { /* ... */ return new(); }
}

// ========== 动画组件 ==========
public class AnimatorComponent : Component
{
    private Dictionary<string, AnimationClip> clips = new();
    private string currentClip = "";
    private float currentTime = 0f;

    public override void Awake()
    {
        // 加载动画资源
        clips["Idle"] = LoadClip("idle");
        clips["Run"] = LoadClip("run");
        clips["Attack"] = LoadClip("attack");
        clips["Death"] = LoadClip("death");
    }

    public void Play(string clipName, bool forceRestart = false)
    {
        if (currentClip == clipName && !forceRestart) return;
        currentClip = clipName;
        currentTime = 0f;
    }

    public override void Update(float dt)
    {
        if (string.IsNullOrEmpty(currentClip)) return;
        currentTime += dt;
        // 更新骨骼变换、播放动画帧...
    }

    private AnimationClip LoadClip(string name) { return new AnimationClip(); }
}

// ========== 使用示例 ==========
public class GameSetup
{
    void CreatePlayer()
    {
        var player = new GameObject { Name = "Player" };
        player.AddComponent<HealthComponent>();
        player.AddComponent<InputComponent>();
        player.AddComponent<MoveComponent>();
        player.AddComponent<WeaponComponent>();
        player.AddComponent<AnimatorComponent>();

        // 获得飞行能力：动态添加组件
        if (player.PickedUpPowerUp == "Flying")
        {
            player.AddComponent<FlyingComponent>();
        }

        // 中毒效果：临时移除移动能力
        player.RemoveComponent<MoveComponent>();
        // 恢复后重新添加
        player.AddComponent<MoveComponent>();
    }

    void CreateEnemy()
    {
        var enemy = new GameObject { Name = "Goblin" };
        enemy.AddComponent<HealthComponent>();
        enemy.AddComponent<MoveComponent>();
        enemy.AddComponent<AIComponent>();        // 替代InputComponent
        enemy.AddComponent<WeaponComponent>();
        enemy.AddComponent<LootDropComponent>();  // 死亡时掉落物品
    }

    void CreateTurret()
    {
        var turret = new GameObject { Name = "Turret" };
        turret.AddComponent<WeaponComponent>();   // 只需要武器
        turret.AddComponent<AIComponent>();       // 自动瞄准
        // 不需要 Health、Move、Input
    }
}
```

### 组件间通信模式

组件不应该直接引用其他组件的内部状态，推荐以下通信模式：

```csharp
// 模式1：通过 Owner.GetComponent 查询（最常用）
public class MoveComponent : Component
{
    public override void Update(float dt)
    {
        var input = Owner.GetComponent<InputComponent>();
        if (input != null)
            Owner.Transform.Position += input.MoveDirection * Speed * dt;
    }
}

// 模式2：通过事件回调（解耦最好）
public class HealthComponent : Component
{
    public event Action<float> OnDamaged;

    public void TakeDamage(float amount)
    {
        Current -= amount;
        OnDamaged?.Invoke(amount); // 通知所有关心伤害的组件
    }
}

public class KnockbackComponent : Component
{
    public override void Awake()
    {
        Owner.GetComponent<HealthComponent>().OnDamaged += OnDamaged;
    }

    void OnDamaged(float damage)
    {
        // 被击退
        ApplyKnockback(5f);
    }
}

// 模式3：通过消息总线（最松耦合）
public class DamageDisplayComponent : Component
{
    public override void Awake()
    {
        EventBus.Subscribe<DamageEvent>(OnDamage);
    }

    void OnDamage(DamageEvent evt)
    {
        if (evt.Target == Owner)
            ShowDamageNumber(evt.Amount);
    }
}
```

### 依赖注入式组件

在更复杂的系统中，可以使用属性标注来声明组件依赖：

```csharp
// 依赖注入标注
[RequireComponent(typeof(HealthComponent))]
[RequireComponent(typeof(MoveComponent))]
public class PlayerController : Component
{
    // 框架自动注入，不需要 GetComponent
    [Inject] private HealthComponent health;
    [Inject] private MoveComponent move;
    [Inject] private AnimatorComponent animator;

    public override void Start()
    {
        // health, move, animator 已经被注入
        health.OnDeath += OnPlayerDeath;
    }

    void OnPlayerDeath()
    {
        animator.Play("Death");
        RemoveComponent<MoveComponent>();
    }
}
```

## 方案对比

| 特性 | 纯继承 | 组件模式 | ECS |
|------|--------|---------|-----|
| 添加新能力 | 创建新子类 | 添加组件 | 添加组件+系统 |
| 运行时变化 | 不支持 | 完全支持 | 完全支持 |
| 代码复用 | 通过继承 | 通过组合 | 通过组合 |
| 缓存友好度 | 差 | 差 | 优秀 |
| 调试难度 | 低 | 中 | 高 |
| 学习曲线 | 低 | 中 | 高 |
| 适用规模 | 小型项目 | 中型项目 | 大型项目 |

## 常见陷阱与解决方案

1. **组件间循环依赖**：A组件更新时需要B，B更新时需要A。解决方案：使用LateUpdate或事件系统
2. **组件查询开销**：频繁 GetComponent 导致性能问题。解决方案：缓存组件引用（在 Awake 中获取）
3. **运行时移除陷阱**：在 Update 中移除同实体上的其他组件可能导致异常。解决方案：延迟移除队列
4. **过度拆分**：把所有逻辑拆成组件导致每个组件只有几行代码。解决方案：根据功能域合理划分
5. **组件间数据竞争**：多个组件同时修改同一数据。解决方案：定义清晰的数据所有权

## Unity 实现

```csharp
// Unity 的 MonoBehaviour 就是组件模式
public class PlayerController : MonoBehaviour
{
    // [RequireComponent] 自动添加依赖组件
    private Rigidbody rb;
    private Animator anim;
    private HealthSystem health;

    void Awake()
    {
        // GetComponent 缓存引用
        rb = GetComponent<Rigidbody>();
        anim = GetComponent<Animator>();
        health = GetComponent<HealthSystem>();
    }

    void Update()
    {
        // 输入处理
        float h = Input.GetAxis("Horizontal");
        float v = Input.GetAxis("Vertical");
        anim.SetFloat("Speed", new Vector2(h, v).magnitude);
    }

    void FixedUpdate()
    {
        // 物理移动
        rb.MovePosition(rb.position + moveDirection * speed * Time.fixedDeltaTime);
    }
}
```

## 实际使用案例

- **Unity 的 GameObject-Component 架构** 是组件模式的典范实现，每个 MonoBehaviour 就是一个组件
- **Unreal Engine 的 Actor-Component 系统** 支持蓝图和 C++ 双语言组件
- **《我的世界》** 的实体系统使用组件存储不同类型的实体属性
- **《饥荒》** 的角色能力通过组件动态组合实现不同角色特性
- **Godot 引擎** 使用 Node（节点）系统实现组件模式，每个节点负责单一功能
