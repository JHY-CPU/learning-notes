# 命令模式（Command）

## 核心概念

命令模式将请求（操作）封装为独立的对象，使得你可以用不同的请求参数化客户端、支持请求排队、记录日志、撤销操作。在游戏开发中，命令模式广泛用于输入处理、录像回放、宏系统、网络同步和回合制游戏的行动记录。

### 为什么需要命令模式？

```
原始方式——直接调用：
void OnKeyPress(Key key)
{
    if (key == Key.W) player.Move(Vector3.Forward);
    if (key == Key.Space) player.Jump();
    if (key == Key.MouseLeft) player.Attack();
}

问题：
1. 无法撤销（没有记录"之前的状态"）
2. 无法录制回放（没有记录"做了什么"）
3. 无法网络同步（无法将操作发送给其他玩家）
4. 输入和执行紧耦合

命令模式——封装操作：
void OnKeyPress(Key key)
{
    ICommand cmd = key switch
    {
        Key.W => new MoveCommand(player, Vector3.Forward),
        Key.Space => new JumpCommand(player),
        Key.MouseLeft => new AttackCommand(player, target),
    };
    commandHistory.Execute(cmd);  // 可以记录、撤销、同步
}
```

### 命令接口与具体命令

```csharp
using System;
using System.Collections.Generic;

// ========== 命令接口 ==========
public interface ICommand
{
    void Execute();
    void Undo();
    string Description { get; } // 用于调试和UI显示
}

// ========== 具体命令实现 ==========

public class MoveCommand : ICommand
{
    private readonly Character character;
    private readonly Vector3 direction;
    private readonly float distance;
    private Vector3 previousPosition;

    public string Description => $"移动 {direction}";

    public MoveCommand(Character character, Vector3 direction, float distance = 1f)
    {
        this.character = character;
        this.direction = direction;
        this.distance = distance;
    }

    public void Execute()
    {
        previousPosition = character.Position;
        character.Position += direction.normalized * distance;
    }

    public void Undo()
    {
        character.Position = previousPosition;
    }
}

public class AttackCommand : ICommand
{
    private readonly Character attacker;
    private readonly GameObject target;
    private readonly float damage;
    private float previousHealth;

    public string Description => $"攻击 {target.name}";

    public AttackCommand(Character attacker, GameObject target, float damage)
    {
        this.attacker = attacker;
        this.target = target;
        this.damage = damage;
    }

    public void Execute()
    {
        var targetHealth = target.GetComponent<HealthComponent>();
        previousHealth = targetHealth.Current;
        targetHealth.TakeDamage(damage);
    }

    public void Undo()
    {
        var targetHealth = target.GetComponent<HealthComponent>();
        targetHealth.Current = previousHealth;
    }
}

public class PlaceBuildingCommand : ICommand
{
    private readonly Player player;
    private readonly BuildingType type;
    private readonly Vector3 position;
    private GameObject placedBuilding;

    public string Description => $"放置建筑 {type} @ {position}";

    public PlaceBuildingCommand(Player player, BuildingType type, Vector3 position)
    {
        this.player = player;
        this.type = type;
        this.position = position;
    }

    public void Execute()
    {
        placedBuilding = BuildingFactory.Create(type, position);
        player.Resources -= BuildingCost.GetCost(type);
    }

    public void Undo()
    {
        Object.Destroy(placedBuilding);
        player.Resources += BuildingCost.GetCost(type);
    }
}

public class CastSpellCommand : ICommand
{
    private readonly Character caster;
    private readonly SpellData spell;
    private readonly Vector3 targetPosition;
    private float previousMana;

    public string Description => $"施放 {spell.SpellName}";

    public CastSpellCommand(Character caster, SpellData spell, Vector3 targetPosition)
    {
        this.caster = caster;
        this.spell = spell;
        this.targetPosition = targetPosition;
    }

    public void Execute()
    {
        previousMana = caster.Mana;
        caster.Mana -= spell.ManaCost;
        SpellSystem.CastSpell(spell, caster, targetPosition);
    }

    public void Undo()
    {
        caster.Mana = previousMana;
        // 注意：伤害效果通常不可逆，但资源消耗可以回滚
    }
}
```

### 撤销/重做系统

```csharp
/// <summary>
/// 命令历史管理器——支持多步撤销和重做
/// </summary>
public class CommandHistory
{
    private readonly Stack<ICommand> undoStack = new();
    private readonly Stack<ICommand> redoStack = new();
    private readonly int maxHistorySize;

    // 事件通知
    public event Action<ICommand> OnCommandExecuted;
    public event Action<ICommand> OnCommandUndone;
    public event Action<ICommand> OnCommandRedone;

    public bool CanUndo => undoStack.Count > 0;
    public bool CanRedo => redoStack.Count > 0;
    public int UndoCount => undoStack.Count;
    public int RedoCount => redoStack.Count;

    public CommandHistory(int maxSize = 100)
    {
        maxHistorySize = maxSize;
    }

    /// <summary>
    /// 执行命令并记录到历史
    /// </summary>
    public void Execute(ICommand command)
    {
        command.Execute();
        undoStack.Push(command);
        redoStack.Clear(); // 新操作清空重做栈

        // 限制历史大小
        if (undoStack.Count > maxHistorySize)
        {
            // 移除最老的命令（栈底）
            var temp = new Stack<ICommand>(undoStack.Reverse().Skip(1).Reverse());
            undoStack.Clear();
            foreach (var cmd in temp) undoStack.Push(cmd);
        }

        OnCommandExecuted?.Invoke(command);
    }

    /// <summary>
    /// 撤销最近的一个命令
    /// </summary>
    public bool Undo()
    {
        if (!CanUndo) return false;

        var command = undoStack.Pop();
        command.Undo();
        redoStack.Push(command);

        OnCommandUndone?.Invoke(command);
        return true;
    }

    /// <summary>
    /// 重做最近撤销的命令
    /// </summary>
    public bool Redo()
    {
        if (!CanRedo) return false;

        var command = redoStack.Pop();
        command.Execute();
        undoStack.Push(command);

        OnCommandRedone?.Invoke(command);
        return true;
    }

    /// <summary>
    /// 撤销多步
    /// </summary>
    public void UndoMultiple(int count)
    {
        for (int i = 0; i < count && CanUndo; i++)
            Undo();
    }

    /// <summary>
    /// 获取撤销栈描述（用于UI显示操作历史）
    /// </summary>
    public List<string> GetUndoDescriptions()
    {
        return undoStack.Select(c => c.Description).ToList();
    }

    /// <summary>
    /// 清空历史
    /// </summary>
    public void Clear()
    {
        undoStack.Clear();
        redoStack.Clear();
    }
}
```

### 宏命令（Macro Command）

将多个命令组合为一个命令：

```csharp
/// <summary>
/// 宏命令——将多个命令组合为一个可撤销的操作
/// </summary>
public class MacroCommand : ICommand
{
    private readonly List<ICommand> commands = new();

    public string Description => $"宏命令 ({commands.Count} 个子操作)";

    public void Add(ICommand cmd) => commands.Add(cmd);

    public void Execute()
    {
        foreach (var cmd in commands)
            cmd.Execute();
    }

    public void Undo()
    {
        // 逆序撤销！
        for (int i = commands.Count - 1; i >= 0; i--)
            commands[i].Undo();
    }
}

// 使用示例：一次移动多个单位
public class MoveArmyCommand : ICommand
{
    private readonly MacroCommand macro = new();
    public string Description => "移动军队";

    public MoveArmyCommand(List<Unit> units, Vector3 destination)
    {
        foreach (var unit in units)
        {
            macro.Add(new MoveCommand(unit, destination - unit.Position));
        }
    }

    public void Execute() => macro.Execute();
    public void Undo() => macro.Undo();
}
```

### 输入录制与回放系统

```csharp
/// <summary>
/// 录像系统——记录带时间戳的命令序列，实现完整回放
/// </summary>
public class ReplaySystem
{
    private struct RecordedCommand
    {
        public float Timestamp;
        public int CommandType;    // 命令类型ID（用于序列化）
        public byte[] Payload;     // 命令参数（序列化后的数据）
    }

    private List<RecordedCommand> recording = new();
    private float gameTimer;
    private bool isRecording = false;
    private bool isReplaying = false;
    private int replayIndex;

    /// <summary>
    /// 开始录制
    /// </summary>
    public void StartRecording()
    {
        recording.Clear();
        gameTimer = 0f;
        isRecording = true;
    }

    /// <summary>
    /// 记录一个命令
    /// </summary>
    public void RecordAndExecute(ICommand command)
    {
        if (isRecording)
        {
            recording.Add(new RecordedCommand
            {
                Timestamp = gameTimer,
                CommandType = CommandRegistry.GetId(command.GetType()),
                Payload = command.Serialize()
            });
        }

        command.Execute();
    }

    /// <summary>
    /// 播放录像
    /// </summary>
    public void StartReplay()
    {
        isReplaying = true;
        gameTimer = 0f;
        replayIndex = 0;
    }

    /// <summary>
    /// 每帧调用，执行到当前时间点的所有命令
    /// </summary>
    public void UpdateReplay(float dt)
    {
        if (!isReplaying) return;

        gameTimer += dt;

        while (replayIndex < recording.Count &&
               recording[replayIndex].Timestamp <= gameTimer)
        {
            var recorded = recording[replayIndex];
            var command = CommandRegistry.CreateCommand(
                recorded.CommandType, recorded.Payload);
            command.Execute();
            replayIndex++;
        }

        if (replayIndex >= recording.Count)
            isReplaying = false;
    }

    /// <summary>
    /// 序列化录像数据（用于保存/网络传输）
    /// </summary>
    public byte[] SerializeReplay()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        writer.Write(recording.Count);
        foreach (var cmd in recording)
        {
            writer.Write(cmd.Timestamp);
            writer.Write(cmd.CommandType);
            writer.Write(cmd.Payload.Length);
            writer.Write(cmd.Payload);
        }

        return stream.ToArray();
    }
}
```

### 网络命令同步

```csharp
/// <summary>
/// 网络命令系统——将命令发送给服务器/其他客户端
/// </summary>
public class NetworkCommandSystem
{
    private CommandHistory localHistory;
    private CommandHistory serverHistory;

    /// <summary>
    /// 客户端：执行命令并发送到服务器
    /// </summary>
    public void ClientExecute(ICommand command)
    {
        // 乐观执行：先在本地执行，提升手感
        command.Execute();
        localHistory.Execute(command);

        // 发送到服务器
        NetworkManager.Send(new CommandPacket
        {
            CommandType = CommandRegistry.GetId(command.GetType()),
            Payload = command.Serialize(),
            Timestamp = Time.time
        });
    }

    /// <summary>
    /// 服务器：接收并验证命令，广播给所有客户端
    /// </summary>
    public void ServerReceive(CommandPacket packet)
    {
        var command = CommandRegistry.CreateCommand(
            packet.CommandType, packet.Payload);

        // 服务器端验证命令合法性
        if (ValidateCommand(command))
        {
            command.Execute();
            serverHistory.Execute(command);

            // 广播给所有客户端
            NetworkManager.Broadcast(packet);
        }
    }

    /// <summary>
    /// 其他客户端：接收服务器广播的命令并执行
    /// </summary>
    public void RemoteExecute(CommandPacket packet)
    {
        var command = CommandRegistry.CreateCommand(
            packet.CommandType, packet.Payload);
        command.Execute();
    }

    private bool ValidateCommand(ICommand command)
    {
        // 检查：冷却是否就绪、资源是否充足、目标是否合法等
        return true;
    }
}
```

### 回合制游戏中的命令模式

```csharp
/// <summary>
/// 回合制行动系统——每回合记录所有玩家的行动
/// </summary>
public class TurnBasedActionSystem
{
    private List<ICommand> currentTurnActions = new();
    private List<List<ICommand>> turnHistory = new();

    /// <summary>
    /// 玩家提交本回合的行动
    /// </summary>
    public void SubmitAction(ICommand action)
    {
        currentTurnActions.Add(action);
    }

    /// <summary>
    /// 执行当前回合所有行动
    /// </summary>
    public void ExecuteTurn()
    {
        // 按行动优先级排序（如：先移动后攻击）
        var sorted = currentTurnActions
            .OrderBy(GetActionPriority)
            .ToList();

        foreach (var action in sorted)
            action.Execute();

        turnHistory.Add(new List<ICommand>(sorted));
        currentTurnActions.Clear();
    }

    /// <summary>
    /// 回滚到指定回合
    /// </summary>
    public void RewindToTurn(int turnIndex)
    {
        // 从最新回合开始逆序撤销
        for (int t = turnHistory.Count - 1; t >= turnIndex; t--)
        {
            var actions = turnHistory[t];
            for (int i = actions.Count - 1; i >= 0; i--)
                actions[i].Undo();
        }

        turnHistory.RemoveRange(turnIndex, turnHistory.Count - turnIndex);
    }

    private int GetActionPriority(ICommand cmd)
    {
        if (cmd is MoveCommand) return 0;     // 先移动
        if (cmd is CastSpellCommand) return 1; // 再施法
        if (cmd is AttackCommand) return 2;    // 最后攻击
        return 10;
    }
}
```

## 方案对比

| 特性 | 命令模式 | 直接调用 | 事件系统 | 策略模式 |
|------|---------|---------|---------|---------|
| 撤销/重做 | 原生支持 | 不支持 | 不支持 | 不支持 |
| 序列化 | 容易 | 不适用 | 部分 | 部分 |
| 录制回放 | 原生支持 | 不支持 | 不支持 | 不支持 |
| 内存开销 | 高（每操作一个对象） | 无 | 中 | 中 |
| 实现复杂度 | 中 | 低 | 中 | 中 |

## 常见陷阱与解决方案

1. **Undo 实现困难**：破坏地形、杀死敌人等操作不可逆。解决方案：使用"逆操作"（创建 vs 销毁）或快照机制
2. **内存占用**：大量命令对象消耗内存。解决方案：限制历史大小、使用命令池
3. **执行顺序**：命令间有依赖关系。解决方案：宏命令中明确定义顺序
4. **网络延迟**：网络命令执行顺序不一致。解决方案：使用确定性锁步 + 输入缓冲

## Unity 实现

```csharp
// Unity Undo 系统就是命令模式
public class MyEditorWindow : EditorWindow
{
    void OnGUI()
    {
        if (GUILayout.Button("Move Object"))
        {
            Undo.RecordObject(target, "Move Object"); // 记录Undo
            target.transform.position += Vector3.forward;
        }
    }
}
```

## Unreal Engine 实现

```cpp
// UE 的 Transaction 系统基于命令模式
void UMyEditor::MoveActor(AActor* Actor, FVector NewLocation)
{
    // FScopedTransaction 自动管理撤销
    const FScopedTransaction Transaction(FText::FromString("Move Actor"));
    Actor->Modify(); // 标记可撤销
    Actor->SetActorLocation(NewLocation);
}
```

## 实际使用案例

- **《星际争霸》** 的录像系统记录所有玩家命令实现完整回放
- **《我的世界》** 世界编辑器支持多步撤销/重做
- **Unity 的 `Undo.RecordObject`** 基于命令模式
- **《文明》系列** 的回合制行动系统使用命令模式支持回滚
- **《俄罗斯方块》** 的回放系统记录每帧输入，支持精确复现
- **Adobe Photoshop** 的历史记录面板是命令模式的经典应用
