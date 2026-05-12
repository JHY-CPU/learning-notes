# GameObject与Component模型

## 核心概念

Unity采用纯组件化架构：GameObject本身只是容器，不含任何功能。所有行为通过挂载Component实现。这是一种Entity-Component模式（非ECS），强调组合优于继承。每个GameObject在底层是一个C++对象的C#包装器，Component通过内部指针与C++核心通信。

## 组件化架构详解

```
GameObject (容器)
├── Transform (必须有，空间变换)
├── MeshRenderer (渲染)
├── Collider (碰撞体)
├── Rigidbody (物理)
├── 自定义脚本 (MonoBehaviour)
├── AudioSource (音频)
└── ... (任意数量的Component)

内部存储结构:
GameObject:
  ├── m_Component: Component[] (组件数组)
  ├── m_Name: string
  ├── m_Tag: string
  ├── m_Layer: int
  ├── m_ActiveFlag: bool
  └── m_Transform: Transform* (指向Transform的快速访问指针)
```

每个Component只负责单一功能，通过组合多个Component实现复杂行为。这是经典的设计模式"组合优于继承"在游戏引擎中的应用。

## Entity-Component vs ECS

Unity的传统EC模式与DOTS中的ECS有本质区别：

| 特性 | 传统EC (MonoBehaviour) | ECS (DOTS) |
|------|----------------------|------------|
| 数据存储 | 每个组件独立对象，分散在堆内存 | 按Archetype连续存储在Chunk中 |
| 遍历方式 | 通过GameObject.Find等查找 | System按Query批量处理 |
| 内存布局 | 引用分散，Cache不友好 | 数据密集排列，Cache友好 |
| 多态 | 继承+虚函数 | 接口约束+泛型 |
| 性能 | 适合中小规模 | 适合大规模(10万+实体) |

## Transform层级系统

Transform是每个GameObject必须拥有的组件，管理空间位置、旋转、缩放和父子关系。Transform之间形成一棵树，根节点在Scene根部。

```csharp
public class TransformDeepDive : MonoBehaviour
{
    void Start()
    {
        // ========== 本地坐标 vs 世界坐标 ==========
        transform.localPosition = new Vector3(0, 1, 0);  // 相对父对象的位置
        transform.position = new Vector3(5, 0, 0);        // 世界坐标（自动更新localPosition）

        // 本地旋转 vs 世界旋转
        transform.localRotation = Quaternion.Euler(0, 90, 0);
        transform.rotation = Quaternion.identity;

        // 本地欧拉角 vs 世界欧拉角
        transform.localEulerAngles = new Vector3(0, 45, 0);
        transform.eulerAngles = new Vector3(0, 90, 0);

        // 缩放
        transform.localScale = Vector3.one * 2f;
        // 注意: lossyScale (世界缩放) 是只读的，受父对象缩放影响

        // ========== 空间方向 ==========
        transform.forward = Vector3.forward; // 设置前方向
        transform.right = Vector3.right;     // 设置右方向
        transform.up = Vector3.up;           // 设置上方向

        // ========== 朝向操作 ==========
        transform.LookAt(target.position);           // 面向目标
        transform.LookAt(target.position, Vector3.up); // 指定上方向

        // 朝向移动方向（平滑旋转）
        if (moveDirection != Vector3.zero)
        {
            transform.forward = moveDirection;
            // 或者使用插值平滑旋转:
            // Quaternion targetRot = Quaternion.LookRotation(moveDirection);
            // transform.rotation = Quaternion.Slerp(transform.rotation, targetRot, Time.deltaTime * rotSpeed);
        }

        // ========== 层级操作 ==========
        transform.SetParent(parentTransform);     // 设置父对象
        transform.SetParent(parentTransform, true); // 设置父对象并保持世界坐标
        transform.SetParent(null);                // 移出层级（成为根对象）
        transform.DetachChildren();               // 分离所有子对象

        // ========== 子对象查找 ==========
        Transform child = transform.Find("ChildName");              // 按名称查找（非递归）
        Transform childDeep = transform.Find("Parent/Child");      // 支持路径
        Transform childByIndex = transform.GetChild(0);            // 按索引
        int childCount = transform.childCount;                     // 子对象数量

        // 遍历所有子对象
        foreach (Transform child2 in transform)
        {
            Debug.Log($"子对象: {child2.name}");
        }

        // ========== 坐标变换 ==========
        // 本地坐标转世界坐标
        Vector3 worldPos = transform.TransformPoint(localPos);
        Vector3 worldDir = transform.TransformDirection(localDir);
        Vector3 worldVec = transform.TransformVector(localVec);

        // 世界坐标转本地坐标
        Vector3 localPos2 = transform.InverseTransformPoint(worldPos);
        Vector3 localDir2 = transform.InverseTransformDirection(worldDir);
    }

    // ========== Transform性能优化 ==========
    void Update()
    {
        // 缓存Transform引用，避免频繁访问属性
        // transform属性实际调用GetComponent<Transform>()
        // 虽然Unity已优化（直接返回内部指针），但缓存仍是好习惯
        Vector3 pos = cachedTransform.position;

        // 使用Transform.Translate（在本地坐标系移动）
        transform.Translate(Vector3.forward * Time.deltaTime * speed);

        // 使用Transform.Rotate（绕轴旋转）
        transform.Rotate(Vector3.up, 90f * Time.deltaTime);

        // 使用Transform.RotateAround（绕点旋转）
        transform.RotateAround(pivot.position, Vector3.up, 30f * Time.deltaTime);
    }

    // 子对象世界坐标变换演示
    void TransformHierarchyDemo()
    {
        // 父对象在(5, 0, 0)，子对象localPosition=(1, 0, 0)
        // 子对象的世界坐标 = (6, 0, 0)
        Transform parent = new GameObject("Parent").transform;
        parent.position = new Vector3(5, 0, 0);

        Transform child = new GameObject("Child").transform;
        child.SetParent(parent);
        child.localPosition = new Vector3(1, 0, 0);

        Debug.Log($"子对象世界坐标: {child.position}"); // (6, 0, 0)

        // 如果父对象缩放为2倍
        parent.localScale = Vector3.one * 2f;
        Debug.Log($"父对象缩放后子对象世界坐标: {child.position}"); // (7, 0, 0)
        Debug.Log($"子对象lossyScale: {child.lossyScale}"); // (2, 2, 2)
    }
}
```

## Transform层级性能优化

Transform层级遍历是常见的性能瓶颈：

```csharp
// 性能对比: 递归遍历 vs 缓存方案
public class TransformOptimizer : MonoBehaviour
{
    // 反模式: 每帧递归遍历整个层级
    void BadTraverse(Transform root)
    {
        foreach (Transform child in root)
        {
            // 对每个子对象做操作
            child.position += Vector3.up * Time.deltaTime;
            BadTraverse(child); // 递归，深度越深开销越大
        }
    }

    // 正确做法: 缓存需要更新的Transform列表
    private Transform[] allChildren;

    void Start()
    {
        // 初始化时缓存
        allChildren = GetComponentsInChildren<Transform>();
    }

    void Update()
    {
        // 批量操作缓存的Transform
        for (int i = 0; i < allChildren.Length; i++)
        {
            allChildren[i].position += Vector3.up * Time.deltaTime;
        }
    }

    // 更好的方案: 只更新需要变化的子集
    private List<Transform> activeChildren = new List<Transform>();

    public void RegisterChild(Transform child)
    {
        if (!activeChildren.Contains(child))
            activeChildren.Add(child);
    }

    public void UnregisterChild(Transform child)
    {
        activeChildren.Remove(child);
    }
}
```

## GetComponent系统

```csharp
public class ComponentManager : MonoBehaviour
{
    // ========== 获取组件 ==========

    void Start()
    {
        // 获取组件（最常用）
        Rigidbody rb = GetComponent<Rigidbody>();
        AudioSource audio = GetComponent<AudioSource>();

        // 带安全检查（推荐方式，避免null引用）
        if (TryGetComponent(out Collider col))
        {
            col.isTrigger = true;
        }

        // 获取子对象上的组件
        MeshRenderer childRenderer = GetComponentInChildren<MeshRenderer>();
        // 包含inactive的组件
        MeshRenderer allRenderers = GetComponentInChildren<MeshRenderer>(true);

        // 获取父对象上的组件
        Canvas parentCanvas = GetComponentInParent<Canvas>();

        // 获取所有同类型组件
        Collider[] allColliders = GetComponents<Collider>();

        // ========== 动态添加组件 ==========
        Rigidbody newRb = gameObject.AddComponent<Rigidbody>();
        newRb.mass = 5f;

        // ========== 移除组件 ==========
        Destroy(GetComponent<BoxCollider>()); // 延迟到帧末执行
    }

    // ========== GetComponent性能分析 ==========
    // GetComponent通过C++反射查找组件列表，开销约为:
    // - 缓存后的直接访问: ~0.001ms
    // - GetComponent调用: ~0.01-0.05ms
    // - GetComponentInChildren: ~0.05-0.2ms (取决于子对象数量)
    // - GameObject.Find: ~0.5-5ms (严重不推荐)
}
```

### SendMessage vs 直接引用对比

```csharp
// 反模式: 使用SendMessage（反射调用，开销大且不安全）
public class SendMessageBad : MonoBehaviour
{
    void Update()
    {
        // 慢! 反射查找+调用，无编译时检查
        gameObject.SendMessage("TakeDamage", 10);
        // 或广播到所有子对象（开销更大）
        gameObject.BroadcastMessage("TakeDamage", 10);
    }
}

// 推荐: 直接引用
public class DirectReferenceGood : MonoBehaviour
{
    private HealthComponent healthComp;

    void Start()
    {
        healthComp = GetComponent<HealthComponent>();
    }

    void Update()
    {
        // 快! 直接方法调用
        if (healthComp != null)
            healthComp.TakeDamage(10);
    }
}

// 最佳实践: 使用接口解耦
public interface IDamageable
{
    void TakeDamage(float amount);
}

public class Enemy : MonoBehaviour, IDamageable
{
    public void TakeDamage(float amount) { /* ... */ }
}

// 调用方无需知道具体类型
public class Weapon : MonoBehaviour
{
    void OnTriggerEnter(Collider other)
    {
        IDamageable target = other.GetComponent<IDamageable>();
        target?.TakeDamage(damage);
    }
}
```

## 查找GameObject的方法与性能

```csharp
public class GameObjectFinder : MonoBehaviour
{
    void Start()
    {
        // ========== 各种查找方式 ==========

        // 方式1: GameObject.Find (开销最大，避免使用)
        GameObject player = GameObject.Find("Player");             // 按名称
        // 搜索整个场景的所有GameObject，O(n)复杂度
        // 缺点: 改名即失效，不安全

        // 方式2: 通过标签查找
        GameObject spawn = GameObject.FindGameObjectWithTag("SpawnPoint");
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");

        // 方式3: Inspector赋值（推荐）
        [SerializeField] private Transform playerTransform;

        // 方式4: 单例模式
        GameManager.Instance.PlayerTransform;

        // 方式5: 依赖注入
        // 通过构造函数或属性注入依赖

        // 方式6: Service Locator
        ServiceLocator.GetService<IPlayerService>();
    }
}

// 单例模式实现
public class GameManager : MonoBehaviour
{
    private static GameManager _instance;
    public static GameManager Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<GameManager>();
                if (_instance == null)
                {
                    var go = new GameObject("GameManager");
                    _instance = go.AddComponent<GameManager>();
                }
            }
            return _instance;
        }
    }

    public Transform PlayerTransform { get; set; }

    void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }
        _instance = this;
        DontDestroyOnLoad(gameObject);
    }
}
```

## 对象池系统

对象池避免频繁的Instantiate和Destroy开销：

```csharp
public class ObjectPool : MonoBehaviour
{
    [System.Serializable]
    public class Pool
    {
        public string tag;
        public GameObject prefab;
        public int initialSize = 10;
    }

    public static ObjectPool Instance;

    [SerializeField] private Pool[] pools;
    private Dictionary<string, Queue<GameObject>> poolDictionary;

    void Awake()
    {
        Instance = this;
        poolDictionary = new Dictionary<string, Queue<GameObject>>();

        // 预热对象池
        foreach (var pool in pools)
        {
            var queue = new Queue<GameObject>();
            for (int i = 0; i < pool.initialSize; i++)
            {
                var obj = Instantiate(pool.prefab);
                obj.SetActive(false);
                queue.Enqueue(obj);
            }
            poolDictionary[pool.tag] = queue;
        }
    }

    public GameObject Get(string tag)
    {
        if (!poolDictionary.ContainsKey(tag))
            return null;

        var queue = poolDictionary[tag];
        GameObject obj;

        if (queue.Count > 0)
        {
            obj = queue.Dequeue();
        }
        else
        {
            // 池为空时动态扩容
            var pool = System.Array.Find(pools, p => p.tag == tag);
            obj = Instantiate(pool.prefab);
        }

        obj.SetActive(true);
        return obj;
    }

    public void Return(GameObject obj, string tag, float delay = 0f)
    {
        if (delay > 0)
        {
            StartCoroutine(ReturnDelayed(obj, tag, delay));
            return;
        }

        obj.SetActive(false);
        poolDictionary[tag].Enqueue(obj);
    }

    IEnumerator ReturnDelayed(GameObject obj, string tag, float delay)
    {
        yield return new WaitForSeconds(delay);
        Return(obj, tag);
    }
}

// 使用示例
public class Bullet : MonoBehaviour
{
    [SerializeField] private float lifetime = 3f;

    void OnEnable()
    {
        // 自动返回对象池
        ObjectPool.Instance.Return(gameObject, "Bullet", lifetime);
    }

    void OnTriggerEnter(Collider other)
    {
        // 提前返回对象池
        ObjectPool.Instance.Return(gameObject, "Bullet");
    }
}
```

## 实际游戏案例

### 案例: RPG角色组件化系统

```csharp
// 角色由多个独立组件组成，每个组件负责单一功能
public class RPGCharacter : MonoBehaviour
{
    // 不用继承一个大的基类，而是组合多个组件
    // - HealthComponent (生命值管理)
    // - ManaComponent (法力值管理)
    // - InventoryComponent (背包系统)
    // - EquipmentComponent (装备系统)
    // - CombatComponent (战斗系统)
    // - SkillComponent (技能系统)
    // - MovementComponent (移动系统)
    // - AnimationComponent (动画系统)
}

public class HealthComponent : MonoBehaviour, IDamageable
{
    [SerializeField] private float maxHealth = 100f;
    private float currentHealth;
    public System.Action<float, float> OnHealthChanged; // (current, max)

    void Awake() { currentHealth = maxHealth; }

    public void TakeDamage(float amount)
    {
        currentHealth = Mathf.Max(0, currentHealth - amount);
        OnHealthChanged?.Invoke(currentHealth, maxHealth);
        if (currentHealth <= 0) Die();
    }

    public void Heal(float amount)
    {
        currentHealth = Mathf.Min(maxHealth, currentHealth + amount);
        OnHealthChanged?.Invoke(currentHealth, maxHealth);
    }

    void Die() { /* 死亡处理 */ }
}
```

### 案例: 场景内对象批量管理

```csharp
public class SceneObjectManager : MonoBehaviour
{
    // 按标签缓存所有对象，避免频繁Find调用
    private Dictionary<string, List<GameObject>> tagCache;

    void Awake()
    {
        tagCache = new Dictionary<string, List<GameObject>>();
    }

    public void CacheAllWithTag(string tag)
    {
        tagCache[tag] = new List<GameObject>(
            GameObject.FindGameObjectsWithTag(tag)
        );
    }

    public List<GameObject> GetByTag(string tag)
    {
        if (!tagCache.ContainsKey(tag))
            CacheAllWithTag(tag);
        return tagCache[tag];
    }

    // 场景切换时清空缓存
    void OnDisable()
    {
        tagCache.Clear();
    }
}
```

## 常见陷阱与最佳实践

1. **GetComponent有性能开销**: 应在Start/Awake中缓存结果，不要在Update中频繁调用
2. **不要用GameObject.Find**: 搜索整个场景，性能差且不安全（改名即失效）
3. **父子关系影响Transform**: 子对象的世界坐标受父对象的Transform和Pivot影响，缩放也会影响子对象
4. **Destroy不是立即执行**: 对象在帧末才真正销毁，调用后仍可访问但下一帧消失。需要立即销毁用DestroyImmediate（仅Editor）
5. **组件顺序有时重要**: 同一对象上多个组件的执行顺序可通过Script Execution Order设置
6. **SetActive有开销**: 频繁开关active状态不如使用对象池回收
7. **不要循环依赖**: 两个组件互相引用可能导致初始化顺序问题
8. **对比SendMessage**: SendMessage/BroadcastMessage使用反射，性能差且无编译时检查，应避免使用

## 关键性能对比

| 操作 | 开销 | 建议 |
|------|------|------|
| GetComponent | 中 (0.01-0.05ms) | 缓存到字段 |
| GameObject.Find | 高 (0.5-5ms) | 避免使用 |
| transform.childCount | 低 | 可频繁使用 |
| Instantiate | 中高 (0.5-2ms) | 使用对象池 |
| Destroy | 中 | 使用对象池回收 |
| SendMessage | 高 (0.1-1ms) | 禁止使用 |
| transform访问 | 极低 (已优化) | 缓存更好但非必须 |
| SetActive | 中 | 避免频繁切换 |

## 与其他系统的关联

- **序列化**: Inspector显示的组件属性通过序列化系统持久化
- **Prefab**: Prefab本质上是GameObject+Component组合的模板
- **ECS/DOTS**: Unity的新ECS架构是对传统Component模型的性能升级
- **物理系统**: Rigidbody和Collider是最常用的物理组件
- **渲染系统**: MeshRenderer、Camera、Light等是渲染相关组件
