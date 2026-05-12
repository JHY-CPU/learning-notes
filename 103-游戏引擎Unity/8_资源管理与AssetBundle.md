# 资源管理与AssetBundle

## 核心概念

Unity资源管理涉及资源的加载、引用、卸载和热更新。主要有三种方案：Resources文件夹、AssetBundle和Addressable Asset System。合理选择资源加载方式直接影响内存管理和热更新能力。

## Resources加载

最简单的方式，将资源放在Resources文件夹下，通过路径字符串加载：

```csharp
public class ResourcesExample : MonoBehaviour
{
    void Start()
    {
        // 同步加载
        GameObject prefab = Resources.Load<GameObject>("Prefabs/Enemy");
        Texture2D tex = Resources.Load<Texture2D>("Textures/UI_Icon");
        AudioClip clip = Resources.Load<AudioClip>("Audio/BGM");

        // 加载文件夹下所有资源
        Sprite[] allSprites = Resources.LoadAll<Sprite>("Sprites/Characters");

        // 实例化
        GameObject enemy = Instantiate(prefab);

        // 卸载未使用的资源（释放内存）
        Resources.UnloadUnusedAssets();
    }

    // 异步加载（大资源推荐）
    IEnumerator LoadAsync()
    {
        ResourceRequest request = Resources.LoadAsync<GameObject>("Prefabs/Boss");
        yield return request;

        GameObject boss = request.asset as GameObject;
        Instantiate(boss);
    }
}
```

**Resources限制**: 不能热更新、包体增大、所有Resources资源一并打包。

## AssetBundle

AssetBundle是Unity传统的热更新方案，将资源打包为独立文件，运行时下载加载：

```csharp
public class AssetBundleManager : MonoBehaviour
{
    // 构建AssetBundle（编辑器工具）
    // BuildPipeline.BuildAssetBundles(outputPath, options, target);

    // 从本地或远程加载
    IEnumerator LoadBundle()
    {
        // 从服务器下载
        string url = "https://yourserver.com/assetbundles/enemies";
        UnityWebRequest request = UnityWebRequestAssetBundle.GetAssetBundle(url);
        yield return request.SendWebRequest();

        AssetBundle bundle = DownloadHandlerAssetBundle.GetContent(request);

        // 从Bundle中加载资源
        GameObject prefab = bundle.LoadAsset<GameObject>("Enemy_Soldier");
        Instantiate(prefab);

        // 卸载Bundle（false只卸载包头，true连同加载的资源一起卸载）
        bundle.Unload(false);
    }

    // 依赖管理：加载Manifest处理依赖关系
    IEnumerator LoadWithDependencies()
    {
        // 先加载主Manifest
        AssetBundle manifestBundle = AssetBundle.LoadFromFile(Path.Combine(bundlePath, "AssetBundles"));
        AssetBundleManifest manifest = manifestBundle.LoadAsset<AssetBundleManifest>("AssetBundleManifest");

        // 获取依赖
        string[] dependencies = manifest.GetAllDependencies("enemies");
        foreach (string dep in dependencies)
        {
            AssetBundle depBundle = AssetBundle.LoadFromFile(Path.Combine(bundlePath, dep));
            // 缓存依赖Bundle
        }

        // 最后加载目标Bundle
        AssetBundle mainBundle = AssetBundle.LoadFromFile(Path.Combine(bundlePath, "enemies"));
    }
}
```

## Addressable Asset System

Addressable是Unity推荐的现代资源管理方案，替代AssetBundle：

```csharp
using UnityEngine.AddressableAssets;
using UnityEngine.ResourceManagement.AsyncOperations;

public class AddressableExample : MonoBehaviour
{
    // 通过地址标签加载
    public async void LoadAsset()
    {
        // 异步加载
        AsyncOperationHandle<GameObject> handle =
            Addressables.LoadAssetAsync<GameObject>("Prefabs/Enemy");
        await handle.Task;

        if (handle.Status == AsyncOperationStatus.Succeeded)
        {
            Instantiate(handle.Result);
        }

        // 释放资源
        Addressables.Release(handle);
    }

    // 加载多个资源
    public async void LoadMultiple()
    {
        AsyncOperationHandle<IList<GameObject>> handle =
            Addressables.LoadAssetsAsync<GameObject>("EnemyLabel", obj =>
            {
                // 每加载一个资源回调一次
                Debug.Log($"已加载: {obj.name}");
            });
        await handle.Task;

        foreach (var obj in handle.Result)
        {
            Instantiate(obj);
        }
        Addressables.Release(handle);
    }

    // 从远程下载（支持CDN）
    public async void LoadFromRemote()
    {
        // Addressable自动处理远程/本地路径
        AsyncOperationHandle<Texture2D> handle =
            Addressables.LoadAssetAsync<Texture2D>("RemoteTexture");
        await handle.Task;
    }

    // 下载大小检查
    public async void CheckDownloadSize()
    {
        long size = await Addressables.GetDownloadSizeAsync("MyLabel").Task;
        Debug.Log($"需要下载: {size / 1024f / 1024f:F2} MB");
    }
}
```

## 热更新方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| Resources | 简单直接 | 不能热更、包体大 | 小型项目、原型 |
| AssetBundle | 可热更、灵活 | 管理复杂、依赖处理麻烦 | 传统热更项目 |
| Addressable | 自动管理依赖、支持远程 | 学习曲线 | 中大型项目（推荐） |
| HybridCLR | C#代码热更 | 配置复杂 | 需要代码热更的项目 |

## 常见陷阱与最佳实践

1. **Resources.UnloadUnusedAssets**: 时机很重要，应在场景切换后调用
2. **AssetBundle依赖地狱**: 忽略依赖会导致资源重复加载，内存爆炸
3. **Addressable Group配置**: 合理分组，按场景/功能划分，不要把所有资源放一个Group
4. **内存管理**: 加载的资源必须对应释放，使用引用计数或Addressable的handle
5. **版本控制**: Addressable的Profile用于区分开发/测试/生产环境的远程路径

## 与其他系统的关联

- **场景管理**: SceneManager.LoadScene也属于资源加载
- **Prefab系统**: Prefab是资源管理的主要对象类型
- **Shader**: Shader变体需在ShaderCollection中预编译，否则运行时卡顿
