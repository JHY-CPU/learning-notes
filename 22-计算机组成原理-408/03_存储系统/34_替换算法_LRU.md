# 35_替换算法_LRU

## 核心概念

- **LRU（Least Recently Used）**：最近最少使用替换算法。当Cache组满需要替换时，淘汰**最近最久未被访问**的Cache行。
- **核心思想**：利用**时间局部性**——最近被访问的数据很可能再次被访问，最久没被访问的最可能不再需要。
- **实现方式**：计数器法（每行一个计数器记录距上次访问的时间）或堆栈法（用堆栈记录访问顺序）。
- **性能**：三种实用替换算法中性能最好，无Belady异常，是408**最常考**的替换算法。

## 原理分析

### 工作流程

1. **初始化**：所有Cache行的LRU计数器设为0
2. **命中时**：被访问行的计数器清零，其他行中比该行原计数器小的加1（或重新排序）
3. **缺失且有空行**：新行放入空行，计数器设为0，其他行不变
4. **缺失且组满**：淘汰计数器最大的行（最久未访问），新行计数器设为0

### 计数器实现详解

每个Cache行维护一个LRU计数器：

| 操作 | 被访问行 | 其他行 |
|------|---------|--------|
| 命中 | 计数器→0 | 计数器 < 原被访问行的计数器的行，其计数器+1 |
| 替换 | 新行计数器→0 | 淘汰计数器最大的行，其余不变 |

**示例**：4路组相联，访问序列 A, B, C, D, B, E, A

| 步骤 | 访问 | 行0 | 行1 | 行2 | 行3 | 命中/缺失 | LRU计数器状态 |
|------|------|-----|-----|-----|-----|----------|-------------|
| 1 | A | A | - | - | - | 缺失 | [0,-,-,-] |
| 2 | B | A | B | - | - | 缺失 | [1,0,-,-] |
| 3 | C | A | B | C | - | 缺失 | [2,1,0,-] |
| 4 | D | A | B | C | D | 缺失 | [3,2,1,0] |
| 5 | B | A | B | C | D | **命中** | [3,0,2,1] |
| 6 | E | A | B | E | D | 缺失(替换C) | [3,1,0,2] |
| 7 | A | A | B | E | D | **命中** | [0,2,1,3] |

命中率 = 2/7 ≈ 28.6%

### 堆栈实现

用一个堆栈记录访问顺序：
- 命中：将被访问的块从堆栈中取出，放到栈顶
- 缺失：若栈满，弹出栈底元素（最久未访问），将新块压入栈顶
- 淘汰对象始终是栈底元素

### 硬件实现

对于R路组相联，需要R个计数器：
- 每次访问更新所有计数器
- 比较器选出最大计数器值对应的行进行替换
- 硬件复杂度：$O(R)$

## 直观理解

**生活类比**：衣柜整理
- 最近穿的衣服放在最方便拿的位置
- 长期不穿的衣服被放到最角落
- 需要腾空间时，先把最角落（最久没穿）的衣服拿走

**记忆口诀**：
> "LRU淘汰最近最少用，命中时更新顺序最灵活"

## 代码/模拟

### Python实现LRU替换算法

```python
"""LRU Cache替换算法模拟 - 适用于408考研复习"""

def lru_cache_simulation(access_sequence, num_ways):
    """
    模拟LRU替换算法（计数器实现）
    :param access_sequence: 访问序列
    :param num_ways: 组相联路数（Cache行数）
    :return: 命中次数
    """
    cache = {}          # {块号: LRU计数器值}
    hits = 0

    for i, block in enumerate(access_sequence):
        if block in cache:
            # 命中：被访问行计数器清零，比原值小的行计数器+1
            old_val = cache[block]
            for b in cache:
                if cache[b] < old_val:
                    cache[b] += 1
            cache[block] = 0
            hits += 1
            status = "命中"
        elif len(cache) < num_ways:
            # 缺失且有空行：新行计数器=0
            cache[block] = 0
            status = "调入"
        else:
            # 缺失且满：淘汰计数器最大的行（最久未访问）
            victim = max(cache, key=cache.get)
            del cache[victim]
            # 新行计数器=0，其余行不变
            cache[block] = 0
            status = f"替换{victim}"

        # 按计数器值排序显示，方便理解
        sorted_cache = sorted(cache.items(), key=lambda x: x[1], reverse=True)
        display = [f"{b}(cnt={c})" for b, c in sorted_cache]
        print(f"步骤{i+1}: 访问{block:>2} | {display!s:<35} | {status}")

    print(f"\n命中率 = {hits}/{len(access_sequence)} = {hits/len(access_sequence):.2%}")
    return hits

# 与笔记中的示例一致：4路组相联，序列 A,B,C,D,B,E,A
print("=== LRU算法模拟 (4路组相联) ===")
lru_cache_simulation(['A','B','C','D','B','E','A'], num_ways=4)
```

**预期输出**（与笔记表格一致）：
- 步骤5访问B：命中，B计数器归0
- 步骤6访问E：替换C（C计数器最大=2）
- 步骤7访问A：命中，A计数器归0
- 命中率 = 2/7 ≈ 28.6%

### LRU堆栈实现

```python
def lru_stack_simulation(access_sequence, cache_size):
    """LRU堆栈实现方式 - 更直观"""
    stack = []  # 栈顶=最近使用，栈底=最久未使用
    hits = 0

    for block in access_sequence:
        if block in stack:
            stack.remove(block)  # 从堆栈中取出
            stack.append(block)  # 放到栈顶
            hits += 1
            print(f"  访问{block:>2}: 命中  | 栈={stack}")
        elif len(stack) < cache_size:
            stack.append(block)
            print(f"  访问{block:>2}: 调入  | 栈={stack}")
        else:
            victim = stack.pop(0)  # 弹出栈底（最久未访问）
            stack.append(block)
            print(f"  访问{block:>2}: 替换{victim} | 栈={stack}")

    print(f"命中率 = {hits}/{len(access_sequence)} = {hits/len(access_sequence):.2%}")

print("\n=== LRU堆栈实现 ===")
lru_stack_simulation(['A','B','C','D','B','E','A'], cache_size=4)
```

## 知识关联

### 与FIFO的关键区别

| 对比项 | LRU | FIFO |
|--------|-----|------|
| 命中时 | **更新计数器** | 不更新 |
| 淘汰依据 | 最久未访问 | 最早进入 |
| Belady异常 | **无** | **有** |
| 硬件开销 | 较高（需记录访问历史） | 较低 |
| 性能 | 好 | 较差 |

### 与OPT（最优算法）的关系

- OPT淘汰将来最长时间不被访问的块（需要预知未来）
- LRU是OPT的近似算法，假设"最近不用的将来也不用"
- OPT是理论最优但不可实现，LRU是实际最优且可实现

### 408考点

- 给定访问序列，用LRU画Cache状态表，计算命中率
- LRU计数器更新规则的理解
- LRU与FIFO在同一序列下的命中率对比
- LRU为什么没有Belady异常

### 为什么LRU没有Belady异常？

LRU的替换决策基于"栈性质"：容量为C时被访问的块集合，一定是容量为C-1时被访问块集合的**超集**。因此增大Cache容量，命中率只会增加或不变，不会减少。

## 易错陷阱

### 陷阱一：LRU命中时计数器更新

很多同学会忘记：LRU命中时**其他行的计数器也要更新**。被访问行变为0，比它原值小的行计数器+1。

### 陷阱二：LRU与FIFO计数器的区别

- FIFO计数器只在**新块调入时**更新
- LRU计数器在**每次访问（包括命中）时**都更新

### 陷阱三：计数器取值范围

R路组相联的LRU计数器取值范围为 $0 \sim R-1$，不会超过路数。

### 陷阱四：LRU的近似实现

实际CPU中，完全的LRU硬件开销太大，通常使用**伪LRU（Pseudo-LRU）**，如树形替换策略，但408考试按标准LRU出题。
