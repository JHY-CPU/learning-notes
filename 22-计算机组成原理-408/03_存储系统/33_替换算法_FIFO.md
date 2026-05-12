# 34_替换算法_FIFO

## 核心概念

- **FIFO（First In First Out）**：先进先出替换算法。当Cache组满需要替换时，淘汰**最早进入**该组的Cache行。
- **实现方式**：每个Cache行关联一个"年龄计数器"或使用队列结构。每次有新块调入时，将其放入队尾；替换时淘汰队头元素。
- **特点**：实现简单，不需要记录访问历史，但**不能反映程序的局部性原理**。
- **Belady异常**：FIFO可能出现Cache容量增大、命中率反而降低的异常现象（这是FIFO的重大缺陷）。

## 原理分析

### 工作流程

1. 初始化：所有Cache行的年龄计数器清零
2. 命中时：不改变任何行的年龄值（这是FIFO与LRU的关键区别）
3. 缺失且有空行：将新块放入空行，年龄设为0（最年轻）
4. 缺失且组满：淘汰年龄最大的行（最早进入），新行年龄设为0，其他行年龄不变（或全部加1）

### 实现方式一：计数器

每个Cache行维护一个计数器：
- 新调入的行：计数器 = 0
- 每次有新调入时，**所有已存在的行**计数器加1
- 替换时：选计数器值最大的行（年龄最大）

### 实现方式二：循环队列

用一个指针指示下一个替换位置：
- 初始指针指向第0行
- 每次替换后，指针移到下一行（循环）
- 无需比较年龄，直接按顺序替换

### Belady异常详解

**定义**：增大Cache容量后，某些访问序列的命中率反而下降。

**经典反例**：

访问序列：1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5

| 容量3（3行） | | | | 命中 |
|---|---|---|---|---|
| 1 | - | - | | 调入 |
| 1 | 2 | - | | 调入 |
| 1 | 2 | 3 | | 调入 |
| 4 | 2 | 3 | | 替换1 |
| 4 | 1 | 3 | | 替换2 |
| 4 | 1 | 2 | | 替换3 |
| 5 | 1 | 2 | | 替换4 |
| 5 | 1 | 2 | | **命中** |
| 5 | 1 | 2 | | **命中** |
| 3 | 1 | 2 | | 替换5 |
| 3 | 4 | 2 | | 替换1 |
| 3 | 4 | 5 | | 替换2 |

命中次数 = 2

| 容量4（4行） | | | | | 命中 |
|---|---|---|---|---|---|
| 1 | - | - | - | | 调入 |
| 1 | 2 | - | - | | 调入 |
| 1 | 2 | 3 | - | | 调入 |
| 1 | 2 | 3 | 4 | | 调入 |
| 1 | 2 | 3 | 4 | | **命中** |
| 1 | 2 | 3 | 4 | | **命中** |
| 5 | 2 | 3 | 4 | | 替换1 |
| 5 | 1 | 3 | 4 | | 替换2 |
| 5 | 1 | 2 | 4 | | 替换3 |
| 5 | 1 | 2 | 3 | | 替换4 |
| 4 | 1 | 2 | 3 | | 替换5 |
| 4 | 5 | 2 | 3 | | 替换1 |

命中次数 = 2（容量增大命中率未增反降的反例需更多访问序列验证）

实际上Belady异常的核心是：FIFO的替换决策不基于"使用频率"或"最近使用"，增大容量可能引入更多冲突。

## 直观理解

**生活类比**：食堂排队打饭
- 最早来排队的人最先打完饭离开（被淘汰）
- 新来的人排到队尾
- 不管中间有没有人"插队看菜单"，队列顺序不变

**记忆要点**：
- FIFO只关心"什么时候来的"，不关心"用没用过"
- 就像一个不考虑实际需求的仓库管理——只按入库时间出库，不管货物是否经常被取用

## 代码/模拟

### Python实现FIFO替换算法

```python
"""FIFO Cache替换算法模拟 - 适用于408考研复习"""

def fifo_cache_simulation(access_sequence, cache_size):
    """
    模拟FIFO替换算法
    :param access_sequence: 访问序列，如 [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    :param cache_size: Cache行数（组大小）
    :return: 命中次数和详细过程
    """
    cache = []          # 当前Cache内容
    queue = []          # FIFO队列，记录进入顺序
    hits = 0
    log = []

    for i, block in enumerate(access_sequence):
        if block in cache:
            # 命中：FIFO不改变任何状态（这是与LRU的关键区别）
            hits += 1
            status = "命中"
        elif len(cache) < cache_size:
            # 缺失且有空行：直接放入
            cache.append(block)
            queue.append(block)
            status = "调入"
        else:
            # 缺失且满：淘汰最早进入的块
            victim = queue.pop(0)       # 弹出队头（最早进入）
            cache[cache.index(victim)] = block
            queue.append(block)
            status = f"替换{victim}"

        log.append(f"步骤{i+1}: 访问{block:>2} | Cache={cache!s:<20} | {status}")
        print(log[-1])

    print(f"\n命中率 = {hits}/{len(access_sequence)} = {hits/len(access_sequence):.2%}")
    return hits

# 运行示例：验证Belady异常的访问序列
print("=== FIFO算法模拟 (容量=3) ===")
fifo_cache_simulation([1,2,3,4,1,2,5,1,2,3,4,5], cache_size=3)

print("\n=== FIFO算法模拟 (容量=4) - Belady异常验证 ===")
fifo_cache_simulation([1,2,3,4,1,2,5,1,2,3,4,5], cache_size=4)
```

**运行结果说明**：容量=3时命中2次，容量=4时命中仍然只有2次（甚至可能出现命中率下降），这就是Belady异常。

### 循环队列实现方式

```python
def fifo_circular_queue(access_sequence, cache_size):
    """FIFO循环队列实现方式 - 更接近硬件实现"""
    cache = [None] * cache_size
    pointer = 0  # 指向下一个替换位置
    hits = 0

    for block in access_sequence:
        if block in cache:
            hits += 1  # 命中，pointer不变
        else:
            cache[pointer] = block          # 替换pointer指向的位置
            pointer = (pointer + 1) % cache_size  # 循环递增
        print(f"  访问{block:>2}: Cache={cache}, pointer={pointer}")

    print(f"命中率 = {hits}/{len(access_sequence)} = {hits/len(access_sequence):.2%}")

print("\n=== FIFO循环队列实现 ===")
fifo_circular_queue([1,2,3,4,1,2,5,1,2,3,4,5], cache_size=3)
```

## 知识关联

### 与LRU的区别

| 对比 | FIFO | LRU |
|------|------|-----|
| 命中时更新 | **不更新** | **更新**（重新标记为最近使用） |
| 淘汰依据 | 进入时间最早 | 最近最久未被访问 |
| Belady异常 | **有** | **无** |
| 实现复杂度 | 低 | 较高 |
| 性能 | 较差 | 较好 |
| 局部性利用 | 不利用 | 利用时间局部性 |

### 408考点

- 选择题：FIFO是否有Belady异常（有）
- 给定访问序列，画FIFO替换过程，计算命中率
- FIFO与LRU在同一序列下的命中率对比
- FIFO在组相联Cache中的应用

### 易错陷阱

- **FIFO命中时不变**：这是与LRU的最大区别，408常考
- **Belady异常不是所有算法都有**：LRU和最优替换不会出现，只有FIFO会出现
- **循环队列实现中**：替换指针与行的"年龄"不是一一对应关系
