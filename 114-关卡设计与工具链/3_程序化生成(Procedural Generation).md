# 程序化生成(Procedural Generation)

## 核心概念

程序化生成使用算法自动生成游戏内容，减少手工制作工作量并提供无限的变化可能。它在Roguelike、开放世界、沙盒等类型的游戏中至关重要。

### 常用算法对比

| 算法 | 特点 | 适用场景 | 难度 |
|------|------|---------|------|
| BSP | 矩形房间+走廊 | 地牢、室内 | 简单 |
| Drunkard's Walk | 自然随机洞穴 | 洞穴、地下城 | 简单 |
| Cellular Automata | 模拟侵蚀效果 | 自然地形 | 简单 |
| Perlin Noise | 连续噪声 | 地形高度图 | 中等 |
| WFC | 基于约束 | 城市、像素艺术 | 复杂 |
| L-System | 递归规则 | 植物、建筑 | 中等 |
| Graph Grammar | 图重写 | 关卡结构 | 复杂 |

### 随机性与可控性

程序化生成的核心挑战是在随机性和可控性之间找到平衡：
- **纯随机**：完全不可预测，可能导致不可玩的关卡
- **种子控制**：使用固定种子产生固定结果，便于调试和分享
- **约束生成**：在规则约束下随机，确保生成结果满足游戏性需求
- **后处理**：生成后进行连通性验证、平滑、清理

## 具体实现方法

### BSP地牢生成（完整实现）

```csharp
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// BSP（二叉空间分割）地牢生成器
/// 递归将空间二分，每个叶节点生成一个房间，连接相邻房间形成走廊
/// </summary>
public class BSPDungeon
{
    [System.Serializable]
    public class Room
    {
        public Rect rect;
        public Vector2Int center;
        public List<Room> connectedRooms = new List<Room>();
    }

    public List<Room> rooms = new List<Room>();
    public HashSet<Vector2Int> floorTiles = new HashSet<Vector2Int>();
    public HashSet<Vector2Int> corridorTiles = new HashSet<Vector2Int>();

    private int minWidth, minHeight;
    private int padding;

    public void Generate(int width, int height,
        int minRoomSize = 4, int maxIterations = 5, int padding = 1)
    {
        this.minWidth = minRoomSize;
        this.minHeight = minRoomSize;
        this.padding = padding;

        rooms.Clear();
        floorTiles.Clear();
        corridorTiles.Clear();

        // 递归分割
        List<Rect> leaves = new List<Rect>();
        Split(new Rect(0, 0, width, height), maxIterations, leaves);

        // 在每个叶节点生成房间
        foreach (var leaf in leaves)
        {
            int rw = Random.Range(minRoomSize, (int)leaf.width - padding * 2);
            int rh = Random.Range(minRoomSize, (int)leaf.height - padding * 2);
            int rx = (int)leaf.x + Random.Range(padding, (int)leaf.width - rw - padding);
            int ry = (int)leaf.y + Random.Range(padding, (int)leaf.height - rh - padding);

            Room room = new Room
            {
                rect = new Rect(rx, ry, rw, rh),
                center = new Vector2Int(rx + rw / 2, ry + rh / 2)
            };
            rooms.Add(room);

            // 标记地板瓦片
            for (int x = rx; x < rx + rw; x++)
            for (int y = ry; y < ry + rh; y++)
                floorTiles.Add(new Vector2Int(x, y));
        }

        // 连接相邻房间
        ConnectRooms();
    }

    void Split(Rect area, int depth, List<Rect> leaves)
    {
        if (depth <= 0 || area.width < minWidth * 2 + padding * 2
            || area.height < minHeight * 2 + padding * 2)
        {
            leaves.Add(area);
            return;
        }

        bool splitHoriz = area.width > area.height;
        if (Mathf.Approximately(area.width, area.height))
            splitHoriz = Random.value > 0.5f;

        if (splitHoriz)
        {
            float splitPos = Random.Range(0.4f, 0.6f) * area.width;
            Split(new Rect(area.x, area.y, splitPos, area.height), depth - 1, leaves);
            Split(new Rect(area.x + splitPos, area.y,
                area.width - splitPos, area.height), depth - 1, leaves);
        }
        else
        {
            float splitPos = Random.Range(0.4f, 0.6f) * area.height;
            Split(new Rect(area.x, area.y, area.width, splitPos), depth - 1, leaves);
            Split(new Rect(area.x, area.y + splitPos,
                area.width, area.height - splitPos), depth - 1, leaves);
        }
    }

    void ConnectRooms()
    {
        for (int i = 1; i < rooms.Count; i++)
        {
            Room a = rooms[i - 1];
            Room b = rooms[i];
            a.connectedRooms.Add(b);
            b.connectedRooms.Add(a);
            CarveCorridor(a.center, b.center);
        }
    }

    void CarveCorridor(Vector2Int from, Vector2Int to)
    {
        Vector2Int pos = from;

        // L形走廊
        while (pos.x != to.x)
        {
            floorTiles.Add(pos);
            corridorTiles.Add(pos);
            pos.x += (to.x > pos.x) ? 1 : -1;
        }
        while (pos.y != to.y)
        {
            floorTiles.Add(pos);
            corridorTiles.Add(pos);
            pos.y += (to.y > pos.y) ? 1 : -1;
        }
    }
}
```

### Drunkard's Walk洞穴生成（完整版）

```csharp
using UnityEngine;

/// <summary>
/// 醉汉行走算法生成自然洞穴
/// 从起点随机方向行走，走过的路径形成洞穴
/// </summary>
public class DrunkardWalk
{
    public bool[,] Generate(int width, int height, int steps,
        int drunkardCount = 1, float fillPercent = 0.4f)
    {
        bool[,] map = new bool[width, height];
        int targetFloor = (int)(width * height * fillPercent);
        int floorCount = 0;

        for (int d = 0; d < drunkardCount; d++)
        {
            int x = width / 2;
            int y = height / 2;

            Vector2Int[] dirs = {
                Vector2Int.up, Vector2Int.down,
                Vector2Int.left, Vector2Int.right
            };

            for (int i = 0; i < steps && floorCount < targetFloor; i++)
            {
                if (!map[x, y])
                {
                    map[x, y] = true;
                    floorCount++;
                }

                Vector2Int dir = dirs[Random.Range(0, 4)];
                x = Mathf.Clamp(x + dir.x, 1, width - 2);
                y = Mathf.Clamp(y + dir.y, 1, height - 2);
            }
        }

        return map;
    }

    /// <summary>
    /// 使用元胞自动机平滑洞穴
    /// 消除小洞和孤立区域
    /// </summary>
    public bool[,] Smooth(bool[,] map, int iterations = 4)
    {
        int w = map.GetLength(0);
        int h = map.GetLength(1);
        bool[,] result = (bool[,])map.Clone();

        for (int iter = 0; iter < iterations; iter++)
        {
            for (int x = 1; x < w - 1; x++)
            for (int y = 1; y < h - 1; y++)
            {
                int neighbors = CountFloorNeighbors(result, x, y);
                if (neighbors > 4)
                    result[x, y] = true;
                else if (neighbors < 4)
                    result[x, y] = false;
            }
        }

        return result;
    }

    int CountFloorNeighbors(bool[,] map, int cx, int cy)
    {
        int count = 0;
        for (int x = cx - 1; x <= cx + 1; x++)
        for (int y = cy - 1; y <= cy + 1; y++)
        {
            if (x == cx && y == cy) continue;
            if (map[x, y]) count++;
        }
        return count;
    }
}
```

### Perlin噪声地形生成

```csharp
using UnityEngine;

/// <summary>
/// Perlin噪声地形生成器
/// 生成连续、自然的高度图
/// </summary>
public class PerlinTerrain
{
    /// <summary>
    /// 生成高度图
    /// </summary>
    public float[,] Generate(int width, int height,
        float scale, int octaves, float persistence, float lacunarity, int seed)
    {
        float[,] heightMap = new float[width, height];

        System.Random rng = new System.Random(seed);
        Vector2[] octaveOffsets = new Vector2[octaves];
        for (int i = 0; i < octaves; i++)
        {
            octaveOffsets[i] = new Vector2(
                rng.Next(-10000, 10000),
                rng.Next(-10000, 10000));
        }

        float maxNoise = float.MinValue;
        float minNoise = float.MaxValue;

        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            float amplitude = 1;
            float frequency = 1;
            float noiseHeight = 0;

            for (int i = 0; i < octaves; i++)
            {
                float sampleX = (x + octaveOffsets[i].x) / scale * frequency;
                float sampleY = (y + octaveOffsets[i].y) / scale * frequency;

                float perlinValue = Mathf.PerlinNoise(sampleX, sampleY) * 2 - 1;
                noiseHeight += perlinValue * amplitude;

                amplitude *= persistence;
                frequency *= lacunarity;
            }

            if (noiseHeight > maxNoise) maxNoise = noiseHeight;
            if (noiseHeight < minNoise) minNoise = noiseHeight;

            heightMap[x, y] = noiseHeight;
        }

        // 归一化到0-1
        for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            heightMap[x, y] = Mathf.InverseLerp(minNoise, maxNoise, heightMap[x, y]);

        return heightMap;
    }
}
```

### 种子控制（可重现生成）

```csharp
/// <summary>
/// 种子控制系统
/// 确保相同种子产生相同结果
/// </summary>
public class SeededGenerator
{
    /// <summary>
    /// 带种子的可重现生成
    /// </summary>
    public bool[,] GenerateWithSeed(int width, int height, int seed)
    {
        // 保存当前随机状态
        Random.State oldState = Random.state;

        // 使用固定种子
        Random.InitState(seed);

        // 执行生成
        var walk = new DrunkardWalk();
        bool[,] map = walk.Generate(width, height, 10000);

        // 恢复随机状态
        Random.state = oldState;

        return map;
    }

    /// <summary>
    /// 基于世界坐标的种子
    /// 用于开放世界的分块生成
    /// </summary>
    public int GetChunkSeed(int worldSeed, int chunkX, int chunkY)
    {
        unchecked
        {
            int hash = worldSeed;
            hash = hash * 31 + chunkX;
            hash = hash * 31 + chunkY;
            return hash;
        }
    }
}
```

## 最佳实践

- 使用种子值控制随机生成，方便调试、分享和多人同步
- 生成后进行连通性验证（Flood Fill），确保所有房间可达
- 设置生成参数的合理范围，避免极端结果（过大/过小的房间）
- 对生成结果进行后处理（平滑、清理孤立区域、填充小洞）
- 考虑游戏性约束（入口/出口位置、Boss房位置、关键道具位置）
- 提供生成预览功能，设计师可以快速调整参数
- 对生成时间进行优化，玩家不应等待超过2-3秒

## 常见陷阱与修复

**陷阱1：未验证生成结果的连通性**
- 症状：玩家被困在封闭区域无法到达出口
- 修复：生成后用Flood Fill验证所有关键点可达

**陷阱2：伪随机数种子使用不当**
- 症状：多次调用产生相同结果，或结果不可重现
- 修复：保存/恢复Random.state，或使用System.Random代替Unity.Random

**陷阱3：WFC算法约束不足**
- 症状：生成失败或崩溃（约束矛盾导致无解）
- 修复：确保约束集合是一致的，添加fallback规则

**陷阱4：生成面积过大导致加载时间过长**
- 症状：生成过程卡顿数秒
- 修复：分块生成，使用协程分帧处理

**陷阱5：程序化内容缺乏手工打磨的细节**
- 症状：关卡感觉重复、缺乏惊喜
- 修复：在程序化生成的基础上手动添加关键叙事点和特殊事件
