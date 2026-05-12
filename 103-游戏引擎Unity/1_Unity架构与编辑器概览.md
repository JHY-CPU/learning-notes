# Unity架构与编辑器概览

## 核心概念

Unity引擎采用组件化(Entity-Component)架构，所有游戏对象由GameObject和Component组合而成。编辑器提供可视化开发环境，通过不同视图协同完成场景构建。Unity的底层由C++编写，上层通过C#脚本与引擎交互，这种"托管-非托管"的桥接机制称为MonoBehaviour/IL2CPP运行时。

## 引擎内部架构层次

```
Unity运行时架构:
┌─────────────────────────────────────────┐
│           C# 脚本层 (Mono/IL2CPP)       │
├─────────────────────────────────────────┤
│         脚本桥接层 (Scripting Layer)      │
├──────────┬──────────┬──────────┬────────┤
│ 渲染管线  │ 物理引擎  │ 动画系统  │ 音频  │
│ SRP/URP  │ PhysX    │ Mecanim  │ FMOD │
├──────────┴──────────┴──────────┴────────┤
│           核心引擎层 (C++)               │
│  Scene Management / Object System       │
│  Serialization / Asset Pipeline         │
│  Job System / Native Collections        │
├─────────────────────────────────────────┤
│         平台抽象层 (Platform Layer)       │
│  DirectX / Vulkan / Metal / OpenGL      │
└─────────────────────────────────────────┘
```

## 编辑器核心视图

### Project视图
管理项目中的所有资源文件（脚本、材质、贴图、预制体等）。Assets文件夹下的内容会自动同步到Project视图。Project视图实际上是对Asset Database的可视化展示，底层使用YAML文件存储资源的meta信息（GUID、导入设置等）。

### Hierarchy视图
显示当前场景中所有GameObject的层级树状结构。父子关系决定了Transform的继承和销毁传播。Hierarchy本质上是场景(Scene)中的活跃对象列表，场景数据以`.unity`文件存储（YAML格式）。

### Inspector视图
显示当前选中对象的所有组件和属性，是最核心的编辑区域。Inspector通过Unity的反射和序列化系统动态构建UI，支持自定义Editor脚本扩展。每个可编辑属性背后对应一个`SerializedProperty`。

### Scene视图
可视化编辑场景的空间布局，支持平移、旋转、缩放操作。Scene视图使用独立的编辑器相机渲染场景（不受Game视图Camera影响），支持2D/3D模式切换、各种Gizmo显示和自定义EditorGUI叠加。

### Game视图
预览游戏运行时的实际画面，受Camera组件和渲染设置影响。Game视图渲染使用与最终发布相同的渲染管线，因此是"所见即所得"的预览。

## 资源管线（Asset Pipeline）

Unity的资源管线是编辑器的核心子系统，负责导入、处理和管理所有外部资源：

```
原始资源(.fbx/.png/.wav)
    ↓ [导入器 Importer]
内部资源格式(.meta + 序列化数据)
    ↓ [Asset Database]
GUID引用系统
    ↓ [运行时加载]
内存中的Unity对象
```

### 资源导入流程

```csharp
// 自定义资源导入器
using UnityEngine;
using UnityEditor;
using UnityEditor.AssetImporters;

[ScriptedImporter(1, "myformat")]
public class MyCustomImporter : ScriptedImporter
{
    public override void OnImportAsset(AssetImportContext ctx)
    {
        // 创建主资源
        var root = ScriptableObject.CreateInstance<MyData>();
        ctx.AddObjectToAsset("main", root);
        ctx.SetMainObject(root);

        // 读取源文件
        string fileContent = File.ReadAllText(ctx.assetPath);

        // 可以创建多个子资源
        var texture = new Texture2D(2, 2);
        ctx.AddObjectToAsset("preview", texture);
    }
}
```

### Asset Database原理

- 每个资源有一个`.meta`文件，存储GUID和导入设置
- 资源间的引用通过GUID+FileID定位，不依赖路径
- 移动/重命名资源时必须同时移动`.meta`文件，否则引用断裂
- Asset Database的刷新通过文件系统监控自动触发

## 序列化系统（Serialization）

Unity的序列化系统是Inspector显示和场景保存的基础：

```csharp
public class SerializationExample : MonoBehaviour
{
    // 可序列化的类型（会被Unity序列化系统处理）
    public int health;                        // 基本值类型
    public string playerName;                 // 字符串
    public Vector3 spawnPosition;             // Unity结构体
    public GameObject targetObject;           // Unity对象引用
    public List<int> inventory = new List<int>(); // List<T>
    public int[] scores = new int[10];        // 数组

    // [Serializable]自定义类可序列化
    [System.Serializable]
    public class Stats
    {
        public float attack;
        public float defense;
        public float speed;
    }
    public Stats playerStats;

    // 不会被序列化的情况
    public Dictionary<string, int> lookup;    // Dictionary不可序列化
    private int hiddenValue;                   // private字段（无[SerializeField]）
    [System.NonSerialized] public int tempValue; // 显式排除

    // 强制序列化private字段
    [SerializeField] private int serializedPrivate;

    // 自定义序列化回调
    [System.Serializable]
    public class SaveData : ISerializationCallbackReceiver
    {
        public Dictionary<string, int> data = new Dictionary<string, int>();

        // 序列化前：将Dictionary转为两个List
        [SerializeField] private List<string> keys = new List<string>();
        [SerializeField] private List<int> values = new List<int>();

        public void OnBeforeSerialize()
        {
            keys.Clear();
            values.Clear();
            foreach (var kvp in data)
            {
                keys.Add(kvp.Key);
                values.Add(kvp.Value);
            }
        }

        public void OnAfterDeserialize()
        {
            data.Clear();
            for (int i = 0; i < keys.Count; i++)
                data[keys[i]] = values[i];
        }
    }
}
```

### 序列化规则总结

| 类型 | 是否可序列化 | 条件 |
|------|-------------|------|
| int, float, bool, string | 是 | 基本类型自动支持 |
| Vector2/3/4, Quaternion, Color | 是 | Unity值类型 |
| enum | 是 | 自动支持 |
| 自定义struct/class | 是 | 标记[System.Serializable] |
| List<T> | 是 | T可序列化 |
| Dictionary<K,V> | 否 | 需手动实现序列化 |
| 委托/事件 | 否 | 不可序列化 |
| MonoBehaviour派生类 | 否 | 不能嵌套为序列化字段 |

## Play Mode生命周期

Unity运行时执行顺序涉及场景加载、对象激活和脚本执行的完整流程：

```
进入Play Mode的完整流程:
1. 保存当前场景状态（用于退出Play Mode后恢复）
2. 编译修改过的脚本（如果有）
3. 加载场景（LoadScene）
4. 实例化所有场景中的GameObject
5. 调用所有脚本的Awake()
6. 调用所有脚本的OnEnable()
7. 调用所有脚本的Start()
8. 进入主循环:
   ├── FixedUpdate() (固定时间步)
   ├── 物理模拟
   ├── Update() (每帧)
   ├── 协程推进
   ├── LateUpdate()
   ├── 渲染
   └── OnGUI / UI更新
9. 检测到退出Play Mode
10. 销毁所有运行时对象
11. 恢复编辑器场景状态
```

```csharp
// 完整的Play Mode生命周期监控
#if UNITY_EDITOR
using UnityEditor;

[InitializeOnLoad]
public class PlayModeMonitor
{
    static PlayModeMonitor()
    {
        EditorApplication.playModeStateChanged += OnPlayModeChanged;
    }

    static void OnPlayModeStateChanged(PlayModeStateChange state)
    {
        switch (state)
        {
            case PlayModeStateChange.ExitingEditMode:
                Debug.Log("即将进入Play Mode - 保存场景");
                break;
            case PlayModeStateChange.EnteredPlayMode:
                Debug.Log("已进入Play Mode");
                break;
            case PlayModeStateChange.ExitingPlayMode:
                Debug.Log("即将退出Play Mode");
                break;
            case PlayModeStateChange.EnteredEditMode:
                Debug.Log("已退出Play Mode - 恢复场景");
                break;
        }
    }
}
#endif
```

## Prefab系统

Prefab是Unity的核心概念，用于创建可复用的游戏对象模板。底层通过PrefabAsset序列化存储，运行时通过Instantiate克隆。

```csharp
// 在代码中实例化Prefab
public class PrefabSpawner : MonoBehaviour
{
    [SerializeField] private GameObject enemyPrefab;
    [SerializeField] private Transform[] spawnPoints;

    void Start()
    {
        // 基本实例化
        GameObject enemy = Instantiate(enemyPrefab, Vector3.zero, Quaternion.identity);

        // 实例化到指定位置和朝向
        foreach (var point in spawnPoints)
        {
            Instantiate(enemyPrefab, point.position, point.rotation);
        }
    }

    // 从Resources加载并实例化（适合Addressable之前的方式）
    void SpawnFromResources()
    {
        GameObject obj = Instantiate(Resources.Load<GameObject>("Prefabs/Enemy"));
    }

    // 使用对象池避免频繁Instantiate/Destroy
    void SpawnFromPool()
    {
        GameObject enemy = ObjectPool.Instance.Get(enemyPrefab);
        enemy.transform.position = spawnPoints[0].position;
    }
}
```

### Prefab嵌套与变体

- **Prefab Variant**: 基于已有Prefab创建变体，可覆盖部分属性。变体继承基础Prefab的修改，适合同类型角色的不同版本
- **Prefab嵌套**: Prefab内可包含其他Prefab，支持模块化设计。子Prefab的修改会反映到所有引用该Prefab的实例
- **Prefab Overrides**: 实例上的修改以Override形式存储，可以Apply到Prefab或Revert回Prefab默认值

```csharp
// 运行时Prefab变体切换
public class CharacterSkinSwitcher : MonoBehaviour
{
    [SerializeField] private AnimatorOverrideController[] skins;
    private Animator animator;

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    public void SwitchSkin(int index)
    {
        if (index >= 0 && index < skins.Length)
            animator.runtimeAnimatorController = skins[index];
    }
}
```

## Project Settings深度解析

Project Settings包含数十个配置模块，以下是核心模块：

| 模块 | 功能 | 关键设置 |
|------|------|---------|
| Player | 平台构建设置 | Company Name, Product Name, Icon, Resolution |
| Quality | 渲染质量等级 | Shadow距离/分辨率, Anti-Aliasing, VSync Count |
| Graphics | 渲染管线和Shader | SRP Asset, Always Included Shaders |
| Input Manager | 旧版输入轴配置 | Axis名称, 正负键, 死区 |
| Physics/Physics2D | 物理引擎全局设置 | 重力, 默认材质, 层碰撞矩阵 |
| Time | 时间管理 | Fixed Timestep, Maximum Allowed Timestep |
| Script Execution Order | 脚本执行顺序 | 自定义脚本的执行优先级 |
| Tags and Layers | 标签和层定义 | 最多32个Layer, 影响渲染和物理 |
| Audio | 全局音频设置 | Volume, DSP Buffer Size |
| Player Settings | IL2CPP/Mono选择 | Api Compatibility Level, Stripping Level |

```csharp
// 运行时读取和修改Quality Settings
public class QualityManager : MonoBehaviour
{
    void Start()
    {
        // 获取当前质量等级
        int currentLevel = QualitySettings.GetQualityLevel();
        string currentName = QualitySettings.names[currentLevel];

        // 运行时切换质量等级
        QualitySettings.SetQualityLevel(2, true); // 应用更改

        // 动态修改渲染设置
        QualitySettings.shadowDistance = 80f;
        QualitySettings.antiAliasing = 4; // MSAA倍数
        QualitySettings.vSyncCount = 1;   // 1=每帧同步

        // 修改物理设置
        Physics.gravity = new Vector3(0, -15f, 0);
        Physics.defaultSolverIterations = 12;

        // 修改时间设置
        Time.fixedDeltaTime = 0.02f; // 50Hz物理更新
        Time.maximumDeltaTime = 0.1f; // 防止帧率过低时物理爆炸
    }
}
```

## 关键属性表

| 属性 | 说明 |
|------|------|
| GameObject.activeSelf | 对象自身是否激活 |
| GameObject.activeInHierarchy | 对象在层级中是否实际激活（受父对象影响） |
| Transform.hierarchyCount | 层级中的子对象数量 |
| tag | 对象标签，用于分类查找 |
| layer | 对象层级，影响渲染和物理碰撞 |
| Scene.isLoaded | 场景是否已加载完成 |
| hideFlags | 对象的隐藏和销毁保护标志 |

## 实际游戏案例

### 案例：开放世界的流式场景加载

大型开放世界游戏（如原神、塞尔达传说）使用场景流式加载，将世界划分为多个区域：

```csharp
public class WorldStreamLoader : MonoBehaviour
{
    [System.Serializable]
    public class WorldChunk
    {
        public string sceneName;
        public Vector3 center;
        public float loadRadius = 200f;
        public float unloadRadius = 300f;
        [HideInInspector] public bool isLoaded;
    }

    [SerializeField] private WorldChunk[] chunks;
    [SerializeField] private Transform player;

    void Update()
    {
        Vector3 playerPos = player.position;

        foreach (var chunk in chunks)
        {
            float dist = Vector3.Distance(playerPos, chunk.center);

            if (!chunk.isLoaded && dist < chunk.loadRadius)
            {
                SceneManager.LoadSceneAsync(chunk.sceneName, LoadSceneMode.Additive);
                chunk.isLoaded = true;
            }
            else if (chunk.isLoaded && dist > chunk.unloadRadius)
            {
                SceneManager.UnloadSceneAsync(chunk.sceneName);
                chunk.isLoaded = false;
            }
        }
    }
}
```

### 案例：自定义Editor工具加速工作流

```csharp
// 批量Prefab处理工具
#if UNITY_EDITOR
using UnityEditor;

public class PrefabBatchTool : EditorWindow
{
    [MenuItem("Tools/Batch Prefab Processor")]
    static void ShowWindow()
    {
        GetWindow<PrefabBatchTool>("Prefab批处理");
    }

    void OnGUI()
    {
        if (GUILayout.Button("批量添加Box Collider到所有Prefab"))
        {
            string[] prefabPaths = AssetDatabase.FindAssets("t:Prefab")
                .Select(guid => AssetDatabase.GUIDToAssetPath(guid)).ToArray();

            foreach (string path in prefabPaths)
            {
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (prefab.GetComponent<Collider>() == null)
                {
                    GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                    instance.AddComponent<BoxCollider>();
                    PrefabUtility.ApplyPrefabInstance(instance, InteractionMode.AutomatedAction);
                    DestroyImmediate(instance);
                }
            }
            AssetDatabase.Refresh();
        }
    }
}
#endif
```

## 常见陷阱与最佳实践

1. **不要在运行时修改Prefab资源**: 运行时对Prefab的修改不会保存，应使用ScriptableObject存储数据
2. **避免过度嵌套Prefab**: 嵌套层级过深会导致Override管理复杂，修改传播不可预测
3. **合理使用Prefab Variant**: 比直接复制Prefab更易于维护，继承关系清晰
4. **Project窗口保持整洁**: 使用合理的文件夹结构，按功能模块分组
5. **序列化陷阱**: Dictionary不可序列化，需要时使用ISerializationCallbackReceiver或自定义方案
6. **Play Mode数据丢失**: 进入Play Mode后对Inspector的修改在退出时会丢失，除非使用`[ExecuteInEditMode]`或Editor脚本
7. **资源引用断裂**: 删除或移动资源时必须通过Unity编辑器操作，不要在文件系统中直接操作
8. **Build Settings遗漏**: 新场景必须添加到Build Settings中，否则打包后无法加载

## 性能分析

| 操作 | 代价 | 优化建议 |
|------|------|---------|
| Instantiate | 高（GC + 内存分配） | 使用对象池预加载 |
| Resources.Load | 中（同步阻塞） | 使用Addressable异步加载 |
| SceneManager.LoadScene | 高（场景重建） | 使用LoadSceneMode.Additive |
| AssetDatabase.FindAssets | 中高（编辑器搜索） | 缓存结果，使用GUID直接查找 |
| GetComponent | 中 | Start/Awake中缓存 |
| 序列化大对象 | 高 | 拆分数据，使用ScriptableObject |

## 与其他系统的关联

- **序列化系统**: Inspector中显示的属性依赖Unity的序列化机制
- **资源管线**: Prefab依赖Asset Database进行资源引用管理
- **场景管理**: Prefab通过SceneManager加载和卸载
- **渲染管线**: URP/HDRP是基于SRP的可编程渲染管线
- **DOTS/ECS**: Unity面向数据的技术栈是对传统GameObject架构的性能升级
