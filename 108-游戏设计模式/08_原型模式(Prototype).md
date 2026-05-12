# 原型模式（Prototype）

## 核心概念

原型模式通过复制现有实例来创建新对象，而非通过类构造函数和 `new` 关键字。在游戏开发中，Prefab（预制体）本质上就是原型模式的应用——预先配置好的对象模板，实例化时通过克隆创建新实例。

### 为什么需要原型模式？

```
传统方式创建敌人：
var goblin = new GameObject();
goblin.AddComponent<HealthComponent>().Max = 50;
goblin.AddComponent<MoveComponent>().Speed = 3;
goblin.AddComponent<AIComponent>().Behavior = "AggressiveMelee";
goblin.AddComponent<WeaponComponent>().Damage = 10;
goblin.AddComponent<LootDropComponent>().Items = [...];
goblin.AddComponent<AnimatorComponent>().Clips = [...];
// 每个哥布林都要写一遍这段代码...20个哥布林就是20遍

原型模式：
GameObject goblinPrefab = LoadGoblinPrefab(); // 配置好一次
for (int i = 0; i < 20; i++)
{
    GameObject goblin = goblinPrefab.Clone();  // 克隆即可
    goblin.Position = randomSpawnPoints[i];
}
```

### 原型接口与深拷贝

```csharp
using System;
using System.Collections.Generic;
using System.Reflection;

// ========== 原型接口 ==========
public interface IPrototype<T>
{
    T Clone();                   // 浅拷贝（共享引用类型的引用）
    T DeepClone();               // 深拷贝（完全独立的副本）
}

// ========== 游戏对象的完整克隆实现 ==========
public class GameObject : IPrototype<GameObject>
{
    public string Name;
    public Vector3 Position;
    public Quaternion Rotation;
    public bool Active = true;
    public List<Component> Components = new();
    public Dictionary<string, object> Tags = new();

    /// <summary>
    /// 浅拷贝：基本类型复制值，引用类型共享引用
    /// 速度快，但修改克隆体的引用类型字段会影响原型
    /// </summary>
    public GameObject Clone()
    {
        var clone = (GameObject)this.MemberwiseClone();
        clone.Components = new List<Component>(Components.Count);
        foreach (var comp in Components)
            clone.Components.Add(comp.Clone());
        clone.Tags = new Dictionary<string, object>(Tags);
        return clone;
    }

    /// <summary>
    /// 深拷贝：所有数据完全独立
    /// 速度慢，但克隆体和原型完全互不影响
    /// </summary>
    public GameObject DeepClone()
    {
        var clone = new GameObject
        {
            Name = this.Name,
            Position = this.Position,
            Rotation = this.Rotation,
            Active = this.Active,
        };

        // 深拷贝所有组件
        foreach (var comp in Components)
            clone.Components.Add(comp.DeepClone());

        // 深拷贝标签
        foreach (var kvp in Tags)
            clone.Tags[kvp.Key] = DeepCopyValue(kvp.Value);

        return clone;
    }

    private object DeepCopyValue(object value)
    {
        if (value == null) return null;
        if (value is ICloneable cloneable) return cloneable.Clone();
        if (value is string) return value; // 字符串不可变，无需深拷贝
        if (value is ValueType) return value; // 值类型天然独立
        // 对于复杂引用类型，使用序列化或反射
        return JsonUtility.FromJson(JsonUtility.ToJson(value), value.GetType());
    }
}

// ========== 组件的克隆 ==========
public abstract class Component : IPrototype<Component>
{
    public GameObject Owner;
    public bool Enabled = true;

    public virtual Component Clone()
    {
        var clone = (Component)this.MemberwiseClone();
        clone.Owner = null; // 克隆后 Owner 需要重新设置
        return clone;
    }

    public virtual Component DeepClone()
    {
        return Clone(); // 默认行为，子类可覆盖
    }
}

public class HealthComponent : Component
{
    public float Current = 100f;
    public float Max = 100f;
    public float RegenRate = 1f;

    public override Component Clone()
    {
        return new HealthComponent
        {
            Current = this.Current,
            Max = this.Max,
            RegenRate = this.RegenRate,
            Enabled = this.Enabled
        };
    }
}

public class MoveComponent : Component
{
    public float Speed = 5f;
    public float Acceleration = 10f;

    public override Component Clone()
    {
        return new MoveComponent
        {
            Speed = this.Speed,
            Acceleration = this.Acceleration,
            Enabled = this.Enabled
        };
    }
}

public class AIComponent : Component
{
    public string BehaviorTree;
    public float DetectionRange = 15f;
    public float AttackRange = 2f;
    public List<Vector3> PatrolPoints = new();

    public override Component Clone()
    {
        return new AIComponent
        {
            BehaviorTree = this.BehaviorTree,
            DetectionRange = this.DetectionRange,
            AttackRange = this.AttackRange,
            PatrolPoints = new List<Vector3>(this.PatrolPoints), // 列表需要独立拷贝
            Enabled = this.Enabled
        };
    }
}

public class LootDropComponent : Component
{
    public List<LootEntry> DropTable = new();

    public override Component Clone()
    {
        var clone = new LootDropComponent
        {
            Enabled = this.Enabled
        };
        // 深拷贝掉落表
        foreach (var entry in DropTable)
            clone.DropTable.Add(entry.Clone());
        return clone;
    }
}

public class LootEntry
{
    public string ItemId;
    public float DropRate;
    public int MinQuantity;
    public int MaxQuantity;

    public LootEntry Clone() => (LootEntry)this.MemberwiseClone();
}
```

### 数据驱动的原型注册表

```csharp
/// <summary>
/// 原型注册表——管理所有可用的原型模板
/// </summary>
public class PrototypeRegistry
{
    private Dictionary<string, GameObject> prototypes = new();

    /// <summary>
    /// 注册一个原型
    /// </summary>
    public void Register(string id, GameObject prototype)
    {
        prototypes[id] = prototype;
    }

    /// <summary>
    /// 根据ID克隆一个实例
    /// </summary>
    public GameObject Create(string id)
    {
        if (!prototypes.TryGetValue(id, out var prototype))
            throw new KeyNotFoundException($"原型 '{id}' 未注册");

        return prototype.DeepClone();
    }

    /// <summary>
    /// 克隆并自定义修改
    /// </summary>
    public GameObject Create(string id, Action<GameObject> customize)
    {
        var instance = Create(id);
        customize?.Invoke(instance);
        return instance;
    }

    /// <summary>
    /// 从JSON配置文件加载原型
    /// </summary>
    public void LoadFromJson(string jsonContent)
    {
        var configs = JsonUtility.FromJson<EnemyConfigFile>(jsonContent);

        foreach (var config in configs.Enemies)
        {
            var prototype = new GameObject { Name = config.Name };

            prototype.AddComponent<HealthComponent>(c =>
            {
                c.Max = config.Health;
                c.Current = config.Health;
            });

            prototype.AddComponent<MoveComponent>(c =>
            {
                c.Speed = config.Speed;
            });

            prototype.AddComponent<AIComponent>(c =>
            {
                c.BehaviorTree = config.AI;
                c.DetectionRange = config.DetectionRange;
            });

            prototype.AddComponent<LootDropComponent>(c =>
            {
                foreach (var item in config.DropTable)
                    c.DropTable.Add(new LootEntry { ItemId = item.Id, DropRate = item.Rate });
            });

            Register(config.Id, prototype);
        }
    }
}

// JSON配置结构
[System.Serializable]
public class EnemyConfigFile
{
    public List<EnemyConfig> Enemies;
}

[System.Serializable]
public class EnemyConfig
{
    public string Id;
    public string Name;
    public float Health;
    public float Damage;
    public float Speed;
    public float DetectionRange;
    public string AI;
    public List<LootConfig> DropTable;
}

[System.Serializable]
public class LootConfig
{
    public string Id;
    public float Rate;
}
```

### 配置驱动生成

```json
{
  "Enemies": [
    {
      "Id": "goblin_basic",
      "Name": "哥布林战士",
      "Health": 50,
      "Damage": 10,
      "Speed": 3.0,
      "DetectionRange": 12.0,
      "AI": "AggressiveMelee",
      "DropTable": [
        { "Id": "gold_coin", "Rate": 0.8 },
        { "Id": "small_potion", "Rate": 0.3 }
      ]
    },
    {
      "Id": "goblin_archer",
      "Name": "哥布林弓箭手",
      "Health": 35,
      "Damage": 15,
      "Speed": 2.5,
      "DetectionRange": 20.0,
      "AI": "RangedKite",
      "DropTable": [
        { "Id": "gold_coin", "Rate": 0.7 },
        { "Id": "arrow_bundle", "Rate": 0.5 }
      ]
    },
    {
      "Id": "dragon_boss",
      "Name": "远古巨龙",
      "Health": 5000,
      "Damage": 200,
      "Speed": 8.0,
      "DetectionRange": 50.0,
      "AI": "BossPattern",
      "DropTable": [
        { "Id": "legendary_sword", "Rate": 1.0 },
        { "Id": "dragon_scale", "Rate": 1.0 },
        { "Id": "rare_gem", "Rate": 0.5 }
      ]
    }
  ]
}
```

```csharp
// 运行时使用
public class EnemySpawner : MonoBehaviour
{
    private PrototypeRegistry registry;

    void Start()
    {
        registry = new PrototypeRegistry();
        var jsonText = Resources.Load<TextAsset>("enemy_configs").text;
        registry.LoadFromJson(jsonText);
    }

    public GameObject SpawnEnemy(string enemyId, Vector3 position)
    {
        var enemy = registry.Create(enemyId, obj =>
        {
            obj.Position = position;
            // 可选的自定义修改
            var health = obj.GetComponent<HealthComponent>();
            health.Current = health.Max; // 确保满血
        });
        return enemy;
    }
}
```

### 随机属性变异（克隆后修改）

```csharp
/// <summary>
/// 带随机变异的原型克隆——用于暗黑破坏神式的随机装备生成
/// </summary>
public class RandomizedPrototypeSpawner
{
    /// <summary>
    /// 从基础原型克隆并随机化属性
    /// </summary>
    public GameObject CreateRandomizedEnemy(string baseId, int level)
    {
        var enemy = registry.Create(baseId);

        // 根据等级缩放属性
        var health = enemy.GetComponent<HealthComponent>();
        health.Max *= (1f + level * 0.1f);
        health.Current = health.Max;

        var move = enemy.GetComponent<MoveComponent>();
        move.Speed *= (1f + level * 0.02f);

        // 随机变异
        if (UnityEngine.Random.value < 0.3f) // 30%概率精英化
        {
            MakeElite(enemy);
        }

        return enemy;
    }

    /// <summary>
    /// 随机装备生成（暗黑破坏神风格）
    /// </summary>
    public Equipment GenerateRandomEquipment(EquipmentRarity rarity)
    {
        // 从基础装备原型克隆
        var baseEquip = equipmentRegistry.Create("base_sword");

        // 根据稀有度添加随机词缀
        int affixCount = rarity switch
        {
            EquipmentRarity.Common => 0,
            EquipmentRarity.Uncommon => 1,
            EquipmentRarity.Rare => 2,
            EquipmentRarity.Epic => 3,
            EquipmentRarity.Legendary => 4,
            _ => 0
        };

        for (int i = 0; i < affixCount; i++)
        {
            var affix = affixPool.GetRandom();
            baseEquip.Modifiers.Add(affix.Clone());
        }

        return baseEquip;
    }
}
```

## 方案对比

| 方案 | 创建速度 | 内存开销 | 灵活性 | 运行时自定义 | 适用场景 |
|------|---------|---------|--------|------------|---------|
| new + 手动配置 | 快 | 低 | 低 | 不支持 | 简单对象 |
| 原型克隆 | 快 | 中 | 高 | 完全支持 | Prefab、敌人 |
| 序列化/反序列化 | 慢 | 中 | 极高 | 完全支持 | 数据驱动 |
| 反射拷贝 | 慢 | 中 | 极高 | 完全支持 | 通用工具 |
| 工厂模式 | 快 | 低 | 中 | 有限 | 类型确定的对象 |

## 常见陷阱与解决方案

1. **引用共享**：浅拷贝后修改克隆体的引用类型字段影响原型。解决方案：使用深拷贝，或在 Clone 中显式拷贝列表/字典
2. **循环引用**：A引用B，B引用A，深拷贝时无限递归。解决方案：维护已拷贝对象的映射表
3. **性能问题**：大量克隆复杂对象。解决方案：使用原型池（预克隆一批）
4. **克隆不完整**：忘记拷贝某个字段导致 Bug。解决方案：使用代码生成工具自动生成 Clone 方法

## Unity 实现

```csharp
// Unity 的 Instantiate 本质就是原型模式
public class Spawner : MonoBehaviour
{
    [SerializeField] private GameObject prefab; // Prefab 就是原型

    void Spawn()
    {
        // Instantiate 克隆 Prefab 的所有组件和属性
        var instance = Instantiate(prefab, spawnPoint.position, Quaternion.identity);

        // 可以修改克隆体
        instance.GetComponent<HealthComponent>().Max = 100 * level;
    }

    // 克隆后修改属性
    void SpawnWithCustomStats()
    {
        var instance = Instantiate(prefab);
        var weapon = instance.GetComponent<WeaponComponent>();
        weapon.Damage *= 1.5f; // 变异：伤害增加50%
    }
}
```

## Unreal Engine 实现

```cpp
// UE 使用 SpawnActor 基于 C++ 原型/模板系统
void AMySpawner::SpawnEnemy()
{
    FActorSpawnParameters SpawnParams;
    SpawnParams.Template = EnemyTemplate; // 模板即原型

    AEnemy* NewEnemy = GetWorld()->SpawnActor<AEnemy>(
        EnemyClass,
        SpawnLocation,
        SpawnRotation,
        SpawnParams
    );

    // 克隆后可修改
    NewEnemy->Health *= LevelMultiplier;
}
```

## 实际使用案例

- **Unity 的 `Instantiate()`** 本质就是原型模式（克隆 Prefab）
- **Unreal Engine 的 `SpawnActor`** 基于 C++ 原型/模板系统
- **《暗黑破坏神》** 的随机装备系统：从基础原型克隆后随机化属性和词缀
- **《我的世界》** 的方块系统通过原型快速实例化百万个相似方块
- **《魔兽世界》** 的怪物生成系统使用原型模板+等级缩放
- **《以撒的结合》** 的道具系统：每个道具是原型的变体
