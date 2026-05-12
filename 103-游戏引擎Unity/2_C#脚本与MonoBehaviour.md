# C#脚本与MonoBehaviour

## 核心概念

MonoBehaviour是Unity中所有脚本组件的基类，提供了与引擎生命周期绑定的回调函数。所有挂载到GameObject上的脚本都必须继承MonoBehaviour。Unity使用Mono（C#运行时）或IL2CPP（将C#转译为C++再编译）来执行脚本，脚本与引擎核心之间通过Scripting Bridge通信。

## 生命周期函数详解

Unity按固定顺序调用生命周期函数，理解执行顺序对游戏逻辑至关重要：

```
完整执行顺序:
1. 加载阶段:   Awake() → OnEnable()
2. 第一帧:     Start() → FixedUpdate() → Update() → LateUpdate() → 渲染
3. 后续帧:     FixedUpdate() → Update() → LateUpdate() → 渲染 (循环)
4. 禁用/销毁:  OnDisable() → OnDestroy()

注意: 有多个脚本时，每个函数的所有实例先完成，再进入下一阶段
例如: 所有Awake() → 所有OnEnable() → 所有Start()
```

```csharp
public class PlayerController : MonoBehaviour
{
    // 对象实例化后立即调用，早于Start
    // 适合初始化引用和设置
    // 注意: 即使脚本被禁用也会调用Awake
    void Awake()
    {
        Debug.Log("Awake - 对象已创建，组件初始化");
        // 初始化自身引用，不依赖其他组件
        rb = GetComponent<Rigidbody>();
    }

    // OnEnable在Awake之后、Start之前调用
    // 每次脚本被启用时都会调用（包括首次）
    void OnEnable()
    {
        Debug.Log("OnEnable - 脚本已启用");
        // 注册事件监听
        EventManager.OnPlayerHit += TakeDamage;
    }

    // 第一次Update之前调用一次
    // 适合获取其他组件的引用（确保其他对象已初始化）
    void Start()
    {
        Debug.Log("Start - 准备开始游戏逻辑");
        // 此时所有对象的Awake和OnEnable都已执行
        // 可以安全地引用其他对象的组件
        enemySpawner = FindObjectOfType<EnemySpawner>();
    }

    // 每帧调用一次，帧率相关
    // 适合处理输入、游戏逻辑
    // deltaTime: 上一帧到当前帧的时间间隔
    void Update()
    {
        float h = Input.GetAxis("Horizontal");
        transform.Translate(Vector3.right * h * Time.deltaTime * 5f);

        // 输入处理
        if (Input.GetKeyDown(KeyCode.Space))
            Jump();
    }

    // 固定时间步调用，默认0.02秒（50Hz）
    // 适合物理计算，与帧率无关
    // Time.fixedDeltaTime控制此间隔
    void FixedUpdate()
    {
        // 物理力操作应在FixedUpdate中
        rb.AddForce(Vector3.forward * 10f);
        // 注意: 不要在FixedUpdate中处理Input
        // 因为FixedUpdate可能跳帧导致丢失输入
    }

    // Update之后调用
    // 适合相机跟随、需要确保在所有Update之后执行的逻辑
    void LateUpdate()
    {
        // 相机跟随在LateUpdate中，确保玩家已在本帧移动完毕
        camera.transform.position = target.position + offset;
        camera.transform.LookAt(target);
    }

    // OnDisable在组件被禁用时调用（包括对象被销毁前）
    void OnDisable()
    {
        Debug.Log("OnDisable - 组件已禁用");
        // 取消事件监听，防止内存泄漏
        EventManager.OnPlayerHit -= TakeDamage;
    }

    // 对象被销毁时调用
    void OnDestroy()
    {
        Debug.Log("OnDestroy - 对象即将销毁");
        // 清理引用和资源
    }
}
```

### ScriptableObject的生命周期

ScriptableObject是不挂载到GameObject上的数据容器，有自己的生命周期：

```csharp
[CreateAssetMenu(fileName = "NewGameData", menuName = "Game/Data Container")]
public class GameData : ScriptableObject
{
    public float playerSpeed = 5f;
    public int maxHealth = 100;
    public Color defaultColor = Color.white;

    // ScriptableObject的生命周期回调
    void OnEnable()  // 资源加载时调用
    {
        Debug.Log("GameData OnEnable - 数据已加载");
    }

    void OnDisable() // 资源卸载时调用
    {
        Debug.Log("GameData OnDisable - 数据卸载");
    }

    void OnDestroy() // 对象被销毁时调用
    {
        Debug.Log("GameData OnDestroy");
    }

    // 运行时修改并持久化（仅Editor模式下）
    void OnValidate() // Inspector中值改变时调用
    {
        playerSpeed = Mathf.Clamp(playerSpeed, 0.1f, 20f);
    }
}

// 使用ScriptableObject存储运行时数据
public class GameManager : MonoBehaviour
{
    [SerializeField] private GameData gameData;

    void Start()
    {
        // ScriptableObject的修改在运行时不会持久化
        // 适合存储配置数据，不适合存储存档数据
        float speed = gameData.playerSpeed;
    }
}
```

## ScriptableRenderPipeline Hooks

URP/HDRP允许通过SRP回调钩子注入自定义渲染逻辑：

```csharp
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

// 在渲染管线中插入自定义Pass
public class CustomRenderPassFeature : ScriptableRendererFeature
{
    class CustomPass : ScriptableRenderPass
    {
        public override void Execute(ScriptableRenderContext context,
            ref RenderingData renderingData)
        {
            // 获取CommandBuffer
            CommandBuffer cmd = CommandBufferPool.Get("CustomPass");

            // 执行自定义渲染操作
            // 例如: 全屏后处理、自定义光照计算
            // Blit(src, dst, material);

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
    }

    CustomPass m_Pass;

    public override void Create()
    {
        m_Pass = new CustomPass();
        m_Pass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer,
        ref RenderingData renderingData)
    {
        renderer.EnqueuePass(m_Pass);
    }
}

// SRP批处理兼容的自定义着色器数据
[System.Serializable, VolumeComponentMenu("Custom/My Post Process")]
public class MyPostProcessVolume : VolumeComponent, IPostProcessComponent
{
    public ClampedFloatParameter intensity = new ClampedFloatParameter(0f, 0f, 1f);

    public bool IsActive() => intensity.value > 0f;
    public bool IsTileCompatible() => true;
}
```

## Attribute系统详解

Unity提供了丰富的属性(Attribute)来自定义Inspector行为和序列化规则：

```csharp
public class AttributeShowcase : MonoBehaviour
{
    // === 序列化控制 ===
    [SerializeField] private int privateField;      // 序列化private字段
    [HideInInspector] public int hiddenPublic;      // 隐藏public字段
    [NonSerialized] public int notSerialized;       // 排除序列化

    // === Inspector显示 ===
    [Header("角色基础属性")]                         // 分组标题
    [Tooltip("角色的生命值上限")]                    // 鼠标悬停提示
    [Range(1, 100)] public int health = 50;         // 数值滑条
    [Min(0)] public float speed = 5f;               // 最小值限制
    [Multiline(3)] public string description;       // 多行文本框
    [TextArea(2, 5)] public string notes;           // 带范围的文本区域

    // === 条件显示 ===
    public bool showAdvanced;
    [SerializeField, ShowIf("showAdvanced")]        // 需自定义ShowIf属性
    private float advancedValue;

    public enum WeaponType { Sword, Bow, Staff }
    public WeaponType weapon;

    [SerializeField]
    [ShowIfEnumValue("weapon", (int)WeaponType.Sword)]
    private float swordDamage;

    // === 资源引用 ===
    [SerializeField] private GameObject prefabRef;          // 拖拽引用
    [SerializeField] private AudioClip[] soundEffects;      // 数组引用
    [SerializeField, SceneObjectsOnly]                     // 需自定义属性
    private Transform sceneTarget;

    // === 按钮功能（需自定义Editor或第三方插件） ===
    [Button("测试伤害")]                                  // 需自定义属性
    public void TestDamage()
    {
        Debug.Log("受到10点伤害");
    }

    // === 运行时验证 ===
    void OnValidate()
    {
        // Inspector中值改变时调用（仅Editor）
        // 用于自动校正和验证
        health = Mathf.Clamp(health, 1, 100);
        speed = Mathf.Max(speed, 0);
    }

    void Reset()
    {
        // 脚本首次添加到GameObject时调用
        // 用于设置默认值
        health = 100;
        speed = 5f;
    }
}

// 自定义属性示例
public class ShowIfAttribute : PropertyAttribute
{
    public string conditionField;
    public ShowIfAttribute(string field) { conditionField = field; }
}

// 对应的PropertyDrawer需要放在Editor文件夹
/*
[CustomPropertyDrawer(typeof(ShowIfAttribute))]
public class ShowIfDrawer : PropertyDrawer
{
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        ShowIfAttribute showIf = (ShowIfAttribute)attribute;
        SerializedProperty conditionProp = property.serializedObject.FindProperty(showIf.conditionField);
        if (conditionProp.boolValue)
            EditorGUI.PropertyField(position, property, label, true);
    }

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        ShowIfAttribute showIf = (ShowIfAttribute)attribute;
        SerializedProperty conditionProp = property.serializedObject.FindProperty(showIf.conditionField);
        return conditionProp.boolValue ? EditorGUI.GetPropertyHeight(property, label) : 0;
    }
}
*/
```

## async/await在Unity中的使用

Unity 2017+支持C#的async/await，但需注意与Unity主线程的交互：

```csharp
using System.Threading.Tasks;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;

public class AsyncExample : MonoBehaviour
{
    // 基本的异步方法
    async void Start()
    {
        // 异步等待2秒（不阻塞主线程）
        await Task.Delay(2000);
        Debug.Log("2秒后执行");

        // 异步加载资源
        string json = await LoadJsonAsync("data/config.json");
        Debug.Log($"加载完成: {json}");

        // 异步HTTP请求
        string response = await GetRequestAsync("https://api.example.com/data");
        Debug.Log($"API响应: {response}");
    }

    // 异步文件读取
    async Task<string> LoadJsonAsync(string path)
    {
        return await Task.Run(() => System.IO.File.ReadAllText(path));
    }

    // 使用UnityWebRequest的异步请求
    async Task<string> GetRequestAsync(string url)
    {
        using (var request = UnityWebRequest.Get(url))
        {
            var operation = request.SendWebRequest();

            // 等待请求完成（通过TaskCompletionSource桥接Unity协程）
            while (!operation.isDone)
                await Task.Yield();

            if (request.result == UnityWebRequest.Result.Success)
                return request.downloadHandler.text;
            else
                throw new System.Exception(request.error);
        }
    }

    // 异步序列执行
    async Task PerformSequenceAsync()
    {
        Debug.Log("步骤1: 加载场景");
        var loadOp = UnityEngine.SceneManagement.SceneManager.LoadSceneAsync("GameScene");
        while (!loadOp.isDone) await Task.Yield();

        Debug.Log("步骤2: 等待玩家准备");
        await WaitForPlayerReadyAsync();

        Debug.Log("步骤3: 开始游戏");
        StartGame();
    }

    // 异步等待条件
    async Task WaitForPlayerReadyAsync()
    {
        while (!IsPlayerReady)
            await Task.Delay(100); // 每100ms检查一次
    }

    // Cancellation Token支持
    async Task<int> LoadWithTimeoutAsync(string url, int timeoutMs)
    {
        using (var cts = new CancellationTokenSource(timeoutMs))
        {
            try
            {
                return await Task.Run(() => LoadData(url), cts.Token);
            }
            catch (TaskCanceledException)
            {
                Debug.LogWarning("加载超时");
                return -1;
            }
        }
    }
}
```

### async/await vs 协程对比

| 特性 | 协程 | async/await |
|------|------|-------------|
| 异常处理 | try-catch困难 | 正常的try-catch |
| 返回值 | 只能yield return | Task<T>返回具体类型 |
| 嵌套 | 嵌套回调地狱 | 正常顺序代码 |
| 取消 | StopCoroutine | CancellationToken |
| 并行 | 多个StartCoroutine | Task.WhenAll |
| Unity兼容性 | 完全支持 | 需注意线程安全 |
| 性能 | 较轻量 | 有一定开销 |

## Job System集成

Unity Job System允许在多线程中执行计算密集任务：

```csharp
using Unity.Jobs;
using Unity.Collections;
using Unity.Burst;
using UnityEngine;

public class JobSystemExample : MonoBehaviour
{
    // 简单的IJob示例
    struct CalculatePositionsJob : IJob
    {
        [ReadOnly] public NativeArray<float> inputs;
        [WriteOnly] public NativeArray<Vector3> results;
        public float deltaTime;

        public void Execute()
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                float angle = inputs[i] * deltaTime;
                results[i] = new Vector3(
                    Mathf.Cos(angle) * 10f,
                    Mathf.Sin(angle) * 5f,
                    0
                );
            }
        }
    }

    // 并行IJobParallelFor示例
    [BurstCompile]
    struct UpdateParticlesJob : IJobParallelFor
    {
        public NativeArray<Vector3> positions;
        [ReadOnly] public NativeArray<Vector3> velocities;
        public float deltaTime;

        public void Execute(int index)
        {
            positions[index] += velocities[index] * deltaTime;
        }
    }

    // NativeArray在主线程创建
    private NativeArray<float> inputArray;
    private NativeArray<Vector3> resultArray;

    void Start()
    {
        int count = 1000;

        // 分配Native内存（必须在主线程）
        inputArray = new NativeArray<float>(count, Allocator.Persistent);
        resultArray = new NativeArray<Vector3>(count, Allocator.Persistent);

        // 初始化数据
        for (int i = 0; i < count; i++)
            inputArray[i] = Random.Range(0f, Mathf.PI * 2f);

        // 调度Job
        var job = new CalculatePositionsJob
        {
            inputs = inputArray,
            results = resultArray,
            deltaTime = Time.deltaTime
        };

        JobHandle handle = job.Schedule();

        // 并行Job示例
        var parallelJob = new UpdateParticlesJob
        {
            positions = resultArray,
            velocities = inputArray.Select(f => new Vector3(f, 0, 0)).ToNativeArray(Allocator.Temp),
            deltaTime = Time.deltaTime
        };

        // 每帧处理1000个粒子，每个粒子一个工作项
        JobHandle parallelHandle = parallelJob.Schedule(count, 64); // 64为batch size

        // 等待完成（在需要结果时）
        handle.Complete();
        parallelHandle.Complete();

        // 读取结果
        for (int i = 0; i < 10; i++)
            Debug.Log($"Position[{i}]: {resultArray[i]}");
    }

    void OnDestroy()
    {
        // 必须释放Native内存
        if (inputArray.IsCreated) inputArray.Dispose();
        if (resultArray.IsCreated) resultArray.Dispose();
    }
}
```

## 协程(Coroutine)深度解析

协程是Unity中处理异步和延时操作的核心机制，基于C#迭代器(IEnumerator)实现：

```csharp
public class SkillManager : MonoBehaviour
{
    [SerializeField] private float cooldownDuration = 3f;
    private bool isCooldown;
    private Coroutine activeCoroutine;

    // 启动协程
    public void StartCooldown()
    {
        activeCoroutine = StartCoroutine(CooldownRoutine(cooldownDuration));
    }

    // 协程函数返回IEnumerator
    private IEnumerator CooldownRoutine(float duration)
    {
        Debug.Log("冷却开始");
        isCooldown = true;

        // 等待指定秒数（受Time.timeScale影响）
        yield return new WaitForSeconds(duration);

        // 等待指定秒数（不受Time.timeScale影响）
        // yield return new WaitForSecondsRealtime(duration);

        isCooldown = false;
        Debug.Log("冷却结束");
    }

    // 各种yield return类型
    IEnumerator YieldExamples()
    {
        yield return null;                       // 等待下一帧
        yield return new WaitForFixedUpdate();   // 等待下一个FixedUpdate
        yield return new WaitForEndOfFrame();    // 等待帧渲染结束
        yield return new WaitForSeconds(1f);     // 等待1秒
        yield return new WaitForSecondsRealtime(1f); // 真实时间等待1秒
        yield return new WaitUntil(() => health <= 0); // 等待条件满足
        yield return new WaitWhile(() => isPaused);    // 等待条件为false
        yield return StartCoroutine(SubCoroutine());   // 嵌套协程
    }

    // 带返回值的协程（通过回调或共享变量）
    private int loadedScore;

    IEnumerator LoadScoreRoutine()
    {
        // 模拟异步加载
        yield return new WaitForSeconds(1f);
        loadedScore = 100; // 将结果存入成员变量
    }

    // 协程链：顺序执行多个异步操作
    IEnumerator SequenceRoutine()
    {
        yield return StartCoroutine(LoadData());
        yield return StartCoroutine(InitializePlayer());
        yield return StartCoroutine(StartGame());
        Debug.Log("所有初始化完成");
    }

    // 并行协程
    IEnumerator ParallelRoutine()
    {
        // 同时启动多个协程
        Coroutine c1 = StartCoroutine(LoadAudio());
        Coroutine c2 = StartCoroutine(LoadTextures());
        Coroutine c3 = StartCoroutine(LoadModels());

        // 等待所有完成
        yield return c1;
        yield return c2;
        yield return c3;

        Debug.Log("所有资源加载完成");
    }

    // 停止协程
    void CancelSkill()
    {
        if (activeCoroutine != null)
            StopCoroutine(activeCoroutine); // 停止单个协程
    }

    void StopAll()
    {
        StopAllCoroutines(); // 停止当前MonoBehaviour的所有协程
    }

    IEnumerator LoadData() { yield return new WaitForSeconds(0.5f); }
    IEnumerator InitializePlayer() { yield return new WaitForSeconds(0.3f); }
    IEnumerator StartGame() { yield return new WaitForSeconds(0.2f); }
    IEnumerator LoadAudio() { yield return new WaitForSeconds(0.8f); }
    IEnumerator LoadTextures() { yield return new WaitForSeconds(1.0f); }
    IEnumerator LoadModels() { yield return new WaitForSeconds(0.6f); }
}
```

### 协程底层原理

协程基于C#的`IEnumerator`和`yield return`语法糖实现。Unity在每帧的协程推进阶段检查所有活跃协程：
1. 调用`MoveNext()`推进迭代器
2. 检查`Current`的类型（WaitForSeconds、WaitForFixedUpdate等）
3. 根据`Current`类型决定何时再次调用`MoveNext()`

```csharp
// 协程的底层等效伪代码
class CoroutineScheduler
{
    List<IEnumerator> coroutines;

    void Update()
    {
        for (int i = coroutines.Count - 1; i >= 0; i--)
        {
            var routine = coroutines[i];
            object current = routine.Current;

            bool shouldAdvance = false;

            if (current == null)
                shouldAdvance = true; // null = 等待一帧
            else if (current is WaitForSeconds wfs)
                shouldAdvance = wfs.IsDone; // 检查计时器
            else if (current is WaitForFixedUpdate)
                shouldAdvance = isInFixedUpdate; // 等待物理帧

            if (shouldAdvance)
            {
                if (!routine.MoveNext())
                    coroutines.RemoveAt(i); // 协程结束
            }
        }
    }
}
```

## 常见陷阱与最佳实践

1. **Awake vs Start混淆**: Awake用于自身初始化，Start用于依赖其他对象的初始化。Awake即使脚本禁用也会调用
2. **FixedUpdate中不要处理输入**: Input应在Update中获取，FixedUpdate可能跳帧导致丢失输入事件
3. **协程不是多线程**: 协程在主线程执行，阻塞操作仍会卡顿。需要多线程用Job System
4. **不要在Update中频繁new对象**: 会导致GC压力，应使用对象池或缓存
5. **OnDestroy中清理引用**: 避免空引用异常，尤其是事件订阅
6. **OnValidate仅Editor**: OnValidate只在Editor中调用，打包后不会执行
7. **async void陷阱**: async void方法无法被等待和捕获异常，尽量用async Task
8. **NativeArray必须Dispose**: Job System中的NativeCollection必须手动释放，否则内存泄漏
9. **脚本执行顺序**: 通过Project Settings或[DefaultExecutionOrder]属性控制多个脚本的执行先后

## 性能分析

| 操作 | 开销 | 说明 |
|------|------|------|
| GetComponent<>() | 中 | 缓存到字段可消除 |
| new对象（堆分配） | 中高 | Update中避免，使用缓存/对象池 |
| 字符串拼接 | 中 | 使用StringBuilder或缓存 |
| 协程启动/停止 | 低中 | 大量协程有管理开销 |
| SendMessage | 高 | 避免使用，改用直接调用 |
| foreach(List) | 中 | 产生GC，使用for循环 |
| LINQ | 中高 | 产生GC，性能敏感处避免 |

## 与其他系统的关联

- **序列化系统**: SerializeField通过Unity序列化管线与Inspector交互
- **脚本执行顺序**: 可在Project Settings -> Script Execution Order中调整，或使用`[DefaultExecutionOrder]`属性
- **Job System**: 性能敏感逻辑可迁移到Unity Job System或DOTS
- **渲染管线**: SRP回调允许在C#中控制渲染流程
- **Addressable**: 异步资源加载的最佳实践是使用Addressable Asset System
