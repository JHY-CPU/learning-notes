# Lua热更新与xLua

## 核心概念

热更新是在不重新发布应用的情况下更新游戏逻辑和资源。由于iOS禁止JIT且C#编译后的DLL难以替换，主流方案是使用Lua脚本语言实现逻辑热更。xLua是腾讯开源的Lua与C#桥接方案，是Unity Lua热更新的事实标准。

## xLua集成

### 安装步骤
1. 下载xLua: https://github.com/Tencent/xLua
2. 将Assets目录复制到项目中
3. 生成Lua绑定代码: XLua -> Generate Code
4. 配置LuaEnv和LuaLoader

```csharp
using XLua;

public class XLuaManager : MonoBehaviour
{
    private LuaEnv luaEnv;

    void Awake()
    {
        luaEnv = new LuaEnv();

        // 自定义Lua文件加载器（用于加载自定义路径的Lua文件）
        luaEnv.AddLoader((ref string filename) =>
        {
            // 从Resources加载
            string path = "Lua/" + filename;
            TextAsset asset = Resources.Load<TextAsset>(path);
            return asset != null ? asset.bytes : null;
        });

        // 从StreamingAssets或持久化路径加载
        luaEnv.AddLoader((ref string filename) =>
        {
            string filePath = Path.Combine(Application.persistentDataPath, "lua", filename + ".lua");
            if (File.Exists(filePath))
                return File.ReadAllBytes(filePath);
            return null;
        });
    }

    void Start()
    {
        // 执行Lua脚本
        luaEnv.DoString("require 'main'");

        // 执行Lua代码片段
        luaEnv.DoString(@"
            print('Hello from Lua!')
            function add(a, b)
                return a + b
            end
        ");

        // 调用Lua全局函数
        var addFunc = luaEnv.Global.Get<System.Func<int, int, int>>("add");
        int result = addFunc(3, 5);
        Debug.Log($"Lua add result: {result}");
    }

    void Update()
    {
        // 每帧执行Lua的update
        luaEnv.Tick();
    }

    void OnDestroy()
    {
        luaEnv?.Dispose();
    }
}
```

## Lua与C#交互

### C#调用Lua

```csharp
[LuaCallCSharp] // xLua标签，标记需要导出到Lua的类
public class PlayerManager : MonoBehaviour
{
    private LuaEnv luaEnv;
    private LuaTable luaSelf;

    void Start()
    {
        luaEnv = XLuaManager.Instance.LuaEnv;

        // 创建Lua表作为self
        luaSelf = luaEnv.NewTable();

        // 设置元表
        LuaTable meta = luaEnv.NewTable();
        meta.Set("__index", luaEnv.Global);
        luaSelf.SetMetaTable(meta);
        meta.Dispose();

        // 执行Lua脚本
        luaEnv.DoString(@"
            function update(self, dt)
                self.moveSpeed = self.moveSpeed + dt * 0.1
            end
        ", "player_script", luaSelf);
    }

    void Update()
    {
        // 调用Lua函数
        var updateFunc = luaSelf.Get<Action<LuaTable, float>>("update");
        updateFunc?.Invoke(luaSelf, Time.deltaTime);
    }

    // 获取Lua中的值
    void GetLuaValue()
    {
        int hp = luaEnv.Global.Get<int>("playerHP");
        string name = luaEnv.Global.Get<string>("playerName");
        var table = luaEnv.Global.Get<LuaTable>("config");
    }
}
```

### Lua调用C#

```csharp
// 标记为Lua可调用
[LuaCallCSharp]
public class GameHelper
{
    public static void LogMessage(string msg)
    {
        Debug.Log($"[Lua] {msg}");
    }

    public int Add(int a, int b) => a + b;
}

// Lua端调用C#
// local helper = CS.GameHelper()
-- helper:LogMessage("Hello from Lua")
-- local result = helper:Add(10, 20)
-- CS.UnityEngine.Debug.Log("使用Unity API")
```

## 热更新流程

```csharp
public class HotUpdateManager : MonoBehaviour
{
    [SerializeField] private string serverUrl = "https://yourserver.com/hotfix/";

    IEnumerator CheckAndUpdate()
    {
        // 1. 下载版本清单
        UnityWebRequest versionReq = UnityWebRequest.Get(serverUrl + "version.txt");
        yield return versionReq.SendWebRequest();

        string remoteVersion = versionReq.downloadHandler.text;
        string localVersion = PlayerPrefs.GetString("lua_version", "0.0.0");

        if (remoteVersion != localVersion)
        {
            // 2. 下载Lua文件包
            UnityWebRequest fileReq = UnityWebRequest.Get(serverUrl + "lua_files.zip");
            yield return fileReq.SendWebRequest();

            // 3. 解压到持久化路径
            string extractPath = Path.Combine(Application.persistentDataPath, "lua");
            ExtractZip(fileReq.downloadHandler.data, extractPath);

            // 4. 保存版本号
            PlayerPrefs.SetString("lua_version", remoteVersion);

            // 5. 重新加载Lua环境
            XLuaManager.Instance.ReloadLuaEnv();

            Debug.Log("热更新完成");
        }
    }

    void ExtractZip(byte[] data, string path)
    {
        // 使用System.IO.Compression或第三方库解压
        // 解压到persistentDataPath的lua目录
    }
}
```

## xLua标签系统

```csharp
// [LuaCallCSharp] - 标记C#类型导出到Lua
[LuaCallCSharp]
public static class ExportConfig
{
    // 批量导出
    [LuaCallCSharp]
    public static List<Type> LuaCallCSharp => new List<Type>()
    {
        typeof(GameObject),
        typeof(Transform),
        typeof(Rigidbody),
        typeof(Debug),
        typeof(Time),
        typeof(Mathf),
    };
}

// [CSharpCallLua] - 标记Lua类型导出到C#
[CSharpCallLua]
public delegate void LuaAction(int a, string b);
```

## 常见陷阱与最佳实践

1. **GC压力**: 频繁的Lua-C#交互产生大量临时对象，应减少跨语言调用频率
2. **闭包捕获**: Lua闭包引用C#对象时注意生命周期管理
3. **LuaEnv单例化**: 项目中应只有一个LuaEnv实例
4. **文件路径一致性**: 确保LuaLoader的路径在编辑器和真机上一致
5. **IL2CPP兼容**: xLua需在IL2CPP构建前生成Wrap代码

## 与其他系统的关联

- **资源管理**: Lua文件本身也需要热更新管理（Addressable/AssetBundle）
- **UI系统**: 常用Lua实现UI逻辑层，C#只负责底层框架
- **HybridCLR**: 新一代C#热更新方案，可替代Lua方案
