# 对象池（Object Pool）

## 核心概念

对象池通过复用已创建的对象来避免频繁的内存分配和垃圾回收（GC）。在游戏中，子弹、敌人、粒子特效、伤害数字等需要频繁创建销毁的对象是对象池的典型应用。对象池的核心思想是：与其每次都新建对象再销毁，不如预先创建一批，用完归还，下次再用。

### 为什么需要对象池？GC 的代价

在 C#（Unity）中，每次 `new` 都会在堆上分配内存。当垃圾回收器（GC）运行时，会暂停所有线程（Stop-The-World）扫描堆内存并回收无用对象。这个暂停可能达到几毫秒甚至几十毫秒，足以造成掉帧。

```
无对象池时（每帧创建100颗子弹）：
帧1: 分配100个Bullet对象 → 绘制 → 分配的旧对象等待回收
帧2: 分配100个Bullet对象 → 绘制
...
帧60: GC触发！暂停15ms → 掉帧！

有对象池时：
帧1: 从池中取出100个Bullet → 绘制 → 归还100个Bullet
帧2: 从池中取出100个Bullet → 绘制 → 归还100个Bullet
...
帧60: 零GC！帧率平稳！
```

### 泛型对象池完整实现

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// 通用对象池——支持预热、动态扩容、容量限制、强制回收
/// </summary>
public class ObjectPool<T> where T : class
{
    // 可用对象队列
    private readonly Queue<T> available = new();

    // 正在使用中的对象集合
    private readonly HashSet<T> inUse = new();

    // 对象工厂（用于创建新实例）
    private readonly Func<T> createFunc;

    // 对象重置回调（归还时清理状态）
    private readonly Action<T> onGet;
    private readonly Action<T> onReturn;

    // 容量配置
    private readonly int initialSize;
    private readonly int maxSize;
    private int totalCount;

    /// <summary>
    /// 当池需要扩容但已达到最大容量时，是否强制回收最老的对象
    /// </summary>
    public bool ForceRecycle { get; set; } = true;

    public int AvailableCount => available.Count;
    public int InUseCount => inUse.Count;
    public int TotalCount => totalCount;

    public ObjectPool(
        Func<T> createFunc,
        Action<T> onGet = null,
        Action<T> onReturn = null,
        int initialSize = 10,
        int maxSize = 1000)
    {
        this.createFunc = createFunc ?? throw new ArgumentNullException(nameof(createFunc));
        this.onGet = onGet;
        this.onReturn = onReturn;
        this.initialSize = initialSize;
        this.maxSize = maxSize;

        WarmUp();
    }

    /// <summary>
    /// 预热：预先创建初始对象
    /// </summary>
    private void WarmUp()
    {
        for (int i = 0; i < initialSize; i++)
        {
            var obj = createFunc();
            totalCount++;
            onReturn?.Invoke(obj); // 初始状态为"已归还"
            available.Enqueue(obj);
        }
    }

    /// <summary>
    /// 从池中获取一个对象
    /// </summary>
    public T Get()
    {
        T obj;

        if (available.Count > 0)
        {
            // 从池中复用
            obj = available.Dequeue();
        }
        else if (totalCount < maxSize)
        {
            // 动态扩容
            obj = createFunc();
            totalCount++;
        }
        else
        {
            // 池满：强制回收最老的对象
            if (ForceRecycle && inUse.Count > 0)
            {
                obj = inUse.First();
                Return(obj);
                obj = available.Dequeue();
            }
            else
            {
                throw new InvalidOperationException(
                    $"ObjectPool<{typeof(T).Name}> 已达最大容量 {maxSize}，" +
                    $"且 ForceRecycle 未启用");
            }
        }

        inUse.Add(obj);
        onGet?.Invoke(obj);
        return obj;
    }

    /// <summary>
    /// 归还对象到池中
    /// </summary>
    public void Return(T obj)
    {
        if (obj == null) return;
        if (!inUse.Remove(obj))
        {
            // 对象不在使用中，可能是重复归还
            return;
        }

        onReturn?.Invoke(obj);
        available.Enqueue(obj);
    }

    /// <summary>
    /// 归还所有正在使用中的对象
    /// </summary>
    public void ReturnAll()
    {
        var allInUse = inUse.ToList();
        foreach (var obj in allInUse)
        {
            onReturn?.Invoke(obj);
            available.Enqueue(obj);
        }
        inUse.Clear();
    }

    /// <summary>
    /// 清空池并销毁所有对象
    /// </summary>
    public void Clear(Action<T> destroyAction = null)
    {
        foreach (var obj in available)
            destroyAction?.Invoke(obj);
        foreach (var obj in inUse)
            destroyAction?.Invoke(obj);

        available.Clear();
        inUse.Clear();
        totalCount = 0;
    }
}
```

### Unity 组件对象池

```csharp
using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Unity MonoBehaviour 专用对象池（管理 GameObject/Component）
/// </summary>
public class UnityObjectPool<T> where T : Component
{
    private readonly Queue<T> available = new();
    private readonly HashSet<T> inUse = new();
    private readonly T prefab;
    private readonly Transform poolParent;
    private readonly int maxSize;

    public UnityObjectPool(T prefab, int initialSize, int maxSize = 500)
    {
        this.prefab = prefab;
        this.maxSize = maxSize;

        // 创建池的父物体（整理Hierarchy）
        poolParent = new GameObject($"Pool_{typeof(T).Name}").transform;
        poolParent.gameObject.SetActive(false); // 隐藏父物体不影响子物体的SetActive

        WarmUp(initialSize);
    }

    private void WarmUp(int count)
    {
        for (int i = 0; i < count; i++)
        {
            var obj = CreateNew();
            obj.gameObject.SetActive(false);
            available.Enqueue(obj);
        }
    }

    public T Get(Vector3 position, Quaternion rotation)
    {
        T obj;
        if (available.Count > 0)
        {
            obj = available.Dequeue();
        }
        else if (inUse.Count < maxSize)
        {
            obj = CreateNew();
        }
        else
        {
            // 强制回收最老的
            obj = inUse.First();
            Return(obj);
            obj = available.Dequeue();
        }

        obj.transform.SetParent(null);
        obj.transform.position = position;
        obj.transform.rotation = rotation;
        obj.gameObject.SetActive(true);
        inUse.Add(obj);

        // 调用组件的初始化回调
        if (obj is IPoolable poolable)
            poolable.OnPoolGet();

        return obj;
    }

    public void Return(T obj)
    {
        if (obj == null || !inUse.Remove(obj)) return;

        if (obj is IPoolable poolable)
            poolable.OnPoolReturn();

        obj.gameObject.SetActive(false);
        obj.transform.SetParent(poolParent);
        available.Enqueue(obj);
    }

    private T CreateNew()
    {
        var obj = Object.Instantiate(prefab, poolParent);
        return obj;
    }
}

// 可池化对象接口
public interface IPoolable
{
    void OnPoolGet();    // 从池中取出时调用
    void OnPoolReturn(); // 归还到池时调用
}
```

### 实际游戏对象使用示例

```csharp
// ===== 子弹池 =====
public class Bullet : MonoBehaviour, IPoolable
{
    private Vector3 velocity;
    private float lifetime;
    private UnityObjectPool<Bullet> pool; // 引用池，用于自动归还

    public void Init(Vector3 velocity, float lifetime, UnityObjectPool<Bullet> pool)
    {
        this.velocity = velocity;
        this.lifetime = lifetime;
        this.pool = pool;
    }

    void Update()
    {
        transform.position += velocity * Time.deltaTime;
        lifetime -= Time.deltaTime;
        if (lifetime <= 0)
            pool.Return(this); // 超时自动归还
    }

    void OnTriggerEnter(Collider other)
    {
        // 命中处理
        var health = other.GetComponent<HealthComponent>();
        health?.TakeDamage(25f);

        // 特效
        VFXManager.Instance.SpawnHitEffect(transform.position);

        // 归还子弹
        pool.Return(this);
    }

    // IPoolable 实现
    public void OnPoolGet()
    {
        // 重置状态
        lifetime = 3f;
        GetComponent<TrailRenderer>().Clear();
    }

    public void OnPoolReturn()
    {
        // 清理
        velocity = Vector3.zero;
    }
}

public class Gun : MonoBehaviour
{
    [SerializeField] private Bullet bulletPrefab;
    [SerializeField] private float bulletSpeed = 50f;
    private UnityObjectPool<Bullet> bulletPool;

    void Start()
    {
        bulletPool = new UnityObjectPool<Bullet>(bulletPrefab, 100, 500);
    }

    void Fire()
    {
        var bullet = bulletPool.Get(muzzle.position, muzzle.rotation);
        bullet.Init(muzzle.forward * bulletSpeed, 3f, bulletPool);
    }
}

// ===== 粒子特效池 =====
public class PooledParticleSystem : MonoBehaviour, IPoolable
{
    private ParticleSystem ps;
    private UnityObjectPool<PooledParticleSystem> pool;

    void Awake() => ps = GetComponent<ParticleSystem>();

    public void Play(float duration, UnityObjectPool<PooledParticleSystem> pool)
    {
        this.pool = pool;
        ps.Play();
        StartCoroutine(ReturnAfterDelay(duration));
    }

    System.Collections.IEnumerator ReturnAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        pool.Return(this);
    }

    public void OnPoolGet() { }
    public void OnPoolReturn() => ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
}

// ===== 伤害飘字池 =====
public class DamageNumber : MonoBehaviour, IPoolable
{
    private TextMeshPro text;
    private float timer;

    public void Show(Vector3 worldPos, int damage, bool isCritical)
    {
        text.text = damage.ToString();
        text.color = isCritical ? Color.yellow : Color.white;
        text.fontSize = isCritical ? 4f : 2.5f;
        transform.position = worldPos;
        timer = 1f;
    }

    void Update()
    {
        timer -= Time.deltaTime;
        transform.position += Vector3.up * 2f * Time.deltaTime; // 飘升效果
        text.alpha = timer; // 渐隐
    }

    public void OnPoolGet() => timer = 1f;
    public void OnPoolReturn() => text.alpha = 0;
}

// ===== 弹壳池 =====
public class ShellCasing : MonoBehaviour, IPoolable
{
    private Rigidbody rb;

    public void Eject(Vector3 position, Vector3 force)
    {
        transform.position = position;
        rb.AddForce(force, ForceMode.Impulse);
        rb.AddTorque(Random.insideUnitSphere * 10f, ForceMode.Impulse);
        StartCoroutine(ReturnAfterDelay(5f));
    }

    System.Collections.IEnumerator ReturnAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        // 通过全局池管理器归还
        PoolManager.Instance.Return("ShellCasing", this);
    }

    public void OnPoolGet() => rb.linearVelocity = Vector3.zero;
    public void OnPoolReturn() { rb.linearVelocity = Vector3.zero; rb.angularVelocity = Vector3.zero; }
}
```

### 全局池管理器

```csharp
/// <summary>
/// 全局池管理器——统一管理所有类型的对象池
/// </summary>
public class PoolManager : MonoBehaviour
{
    public static PoolManager Instance { get; private set; }

    private Dictionary<string, object> pools = new();

    void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance = this;

        // 注册所有池
        RegisterPool("Bullet", new UnityObjectPool<Bullet>(bulletPrefab, 200, 1000));
        RegisterPool("MuzzleFlash", new UnityObjectPool<MuzzleFlash>(muzzlePrefab, 20, 50));
        RegisterPool("BloodSplatter", new UnityObjectPool<BloodSplatter>(bloodPrefab, 50, 200));
        RegisterPool("ShellCasing", new UnityObjectPool<ShellCasing>(shellPrefab, 50, 200));
        RegisterPool("DamageNumber", new UnityObjectPool<DamageNumber>(numberPrefab, 30, 100));
    }

    public void RegisterPool<T>(string name, UnityObjectPool<T> pool) where T : Component
    {
        pools[name] = pool;
    }

    public T Get<T>(string name, Vector3 position, Quaternion rotation) where T : Component
    {
        return ((UnityObjectPool<T>)pools[name]).Get(position, rotation);
    }

    public void Return<T>(string name, T obj) where T : Component
    {
        ((UnityObjectPool<T>)pools[name]).Return(obj);
    }

    // 场景切换时清空所有池
    public void ClearAll()
    {
        foreach (var pool in pools.Values)
        {
            // 通过反射调用 ReturnAll
            pool.GetType().GetMethod("ReturnAll")?.Invoke(pool, null);
        }
    }
}
```

## 方案对比

| 方案 | GC 压力 | 内存效率 | 实现复杂度 | 适用对象 |
|------|--------|---------|-----------|---------|
| 直接 new/destroy | 高 | 高 | 无 | 长生命周期对象 |
| 基础对象池 | 低 | 中 | 低 | 子弹、特效 |
| 带预热的对象池 | 零 | 中 | 中 | 高频创建对象 |
| 带强制回收的池 | 零 | 高 | 中 | 弹壳、飘字 |
| Unity内置 Pool | 零 | 高 | 低 | 通用（2021+） |

## 常见陷阱与解决方案

1. **状态未重置**：从池中取出的对象保留了上次的脏数据。解决方案：实现 IPoolable 接口，在 OnPoolGet 中重置
2. **归还后仍在引用**：其他代码持有已归还对象的引用并继续使用。解决方案：归还时清空外部引用
3. **池大小不当**：太小导致频繁扩容，太大浪费内存。解决方案：根据实际峰值用量设置
4. **跨场景残留**：DontDestroyOnLoad 的池在场景切换后残留。解决方案：监听场景切换事件清空池
5. **嵌套对象池**：一个池化对象内部包含另一个池化对象（如子弹上的特效）。解决方案：先归还子对象再归还父对象

## Unity 内置对象池（2021+）

```csharp
using UnityEngine.Pool;

// Unity 官方提供的泛型对象池
public class BulletSpawner : MonoBehaviour
{
    [SerializeField] private Bullet prefab;

    // 内置池
    private ObjectPool<Bullet> pool;

    void Awake()
    {
        pool = new ObjectPool<Bullet>(
            createFunc: () => Instantiate(prefab),
            actionOnGet: b => b.gameObject.SetActive(true),
            actionOnRelease: b => b.gameObject.SetActive(false),
            actionOnDestroy: b => Destroy(b.gameObject),
            collectionCheck: true,  // 检测重复归还
            defaultCapacity: 100,
            maxSize: 500
        );
    }

    void Fire()
    {
        var bullet = pool.Get();
        // 使用完后归还
        bullet.OnReturnCallback = () => pool.Release(bullet);
    }
}
```

## 实际使用案例

- **Unity 的 `UnityEngine.Pool.ObjectPool<T>`**（2021+版本内置）
- **《使命召唤》** 的弹壳、弹孔、血迹全部使用对象池，单局游戏零 GC
- **《堡垒之夜》** 的建筑碎片、弹道轨迹大量使用对象池
- **《王者荣耀》** 的技能特效、伤害数字使用对象池管理
- **《原神》** 的元素反应特效、战斗飘字使用分层对象池
