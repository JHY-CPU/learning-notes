# 41_Cache性能分析

## 核心概念

- **Cache性能指标**：平均访存时间（AMAT, Average Memory Access Time）、命中率（Hit Rate）、缺失率（Miss Rate）、缺失代价（Miss Penalty）。
- **平均访存时间公式**：

$$\text{AMAT} = \text{Hit Time} + \text{Miss Rate} \times \text{Miss Penalty}$$

- **影响Cache性能的五大因素**：Cache容量、块大小、相联度、替换算法、写策略。
- **408重点**：理解各因素对命中率和平均访存时间的影响趋势，以及各因素之间的权衡关系。

## 原理分析

### 五大影响因素详解

#### 因素一：Cache容量

| Cache容量 | 命中率 | 访问时间 | 缺失率 |
|-----------|--------|---------|--------|
| 增大 | 提高（减少容量缺失） | 可能增加（查找更慢） | 降低 |

- 容量增大 → 可存放更多数据块 → 容量缺失减少
- 但容量增大会增加命中时间（更大的Cache查找更慢）
- **边际收益递减**：容量增大到一定程度后命中率提升很小

#### 因素二：块大小（Block Size）

$$\text{块数} = \frac{\text{Cache容量}}{\text{块大小}}$$

| 块大小 | 空间局部性利用 | 冲突缺失 | 缺失代价 |
|--------|-------------|---------|---------|
| 增大 | 更好 | 可能增加 | 增加（传输时间长） |

- 块太小：空间局部性利用不足，缺失率高
- 块太大：Cache行数减少，冲突缺失增加；缺失代价增大
- **最优块大小**：通常64B~256B，取决于程序特性和存储器带宽

#### 因素三：相联度

| 相联度 | 冲突缺失 | 命中时间 | 硬件复杂度 |
|--------|---------|---------|-----------|
| 提高 | 减少 | 增加 | 增加 |

- 更高相联度 → 减少冲突缺失 → 命中率提高
- 但需要更多比较器 → 命中时间增加
- **实践选择**：L1 Cache 通常4~8路，L2/L3可能16路或更高

#### 因素四：替换算法

| 替换算法 | 缺失率 | 硬件复杂度 |
|---------|--------|-----------|
| OPT（最优） | 最低 | 不可实现 |
| LRU | 低 | 较高 |
| FIFO | 较高 | 低 |
| 随机 | 较高 | 最低 |

- LRU性能最优但硬件开销大
- 实际中常用伪LRU近似

#### 因素五：写策略

| 写策略 | 写缺失率 | 写时间 | 带宽占用 |
|--------|---------|--------|---------|
| 全写法+非写分配 | 较高 | 慢 | 高 |
| 写回法+写分配 | 较低 | 快 | 低 |

- 写回法减少主存访问，但需要脏位
- 全写法需要写缓冲器配合

### 各因素的综合影响

$$\text{AMAT} = t_h + m \times t_p$$

- $t_h$：命中时间（受容量、相联度影响）
- $m$：缺失率（受容量、块大小、相联度、替换算法影响）
- $t_p$：缺失代价（受块大小、写策略影响）

### 性能权衡关系

```
增大Cache容量 → 命中率↑ 但命中时间↑
增大块大小   → 缺失率↓但缺失代价↑（有最优值）
提高相联度   → 缺失率↓但命中时间↑
使用LRU      → 缺失率↓但硬件复杂度↑
使用写回法   → 写速度↑但一致性管理复杂
```

## 直观理解

**Cache性能像一个"效率仓库"**：
- **容量**：仓库大小，越大能放越多货物
- **块大小**：每次进货的包装单位，太大浪费空间，太小进货频繁
- **相联度**：货架的灵活度，灵活度高但查找慢
- **替换算法**：出货策略，好的策略让常用货留下
- **写策略**：修改记录方式，同步记录或延迟记录

## 知识关联

- 与**第5章CPU性能**联系：CPI中访存指令的执行时间受Cache命中率影响
- 与**多级存储器**联系：L1追求速度（小容量、低相联度），L2/L3追求命中率（大容量、高相联度）
- 与**虚拟存储器**联系：页表相当于一个全相联Cache，TLB是页表的Cache

### 408考点

- 选择题：增大某个因素对性能的影响方向
## 代码/模拟

### Python计算Cache性能指标

```python
"""Cache性能分析计算 - 适用于408考研复习"""

def amat(hit_time, miss_rate, miss_penalty):
    """
    计算平均访存时间 AMAT = HitTime + MissRate × MissPenalty
    """
    return hit_time + miss_rate * miss_penalty

def cache_performance_analysis():
    """Cache性能五大因素分析"""

    print("=== AMAT基本公式 ===")
    print("AMAT = HitTime + MissRate × MissPenalty\n")

    # 示例计算
    print("=== 示例: L1 Cache ===")
    ht = 1       # 命中时间 1ns
    mr = 0.05    # 缺失率 5%
    mp = 50      # 缺失代价 50ns
    result = amat(ht, mr, mp)
    print(f"  HitTime={ht}ns, MissRate={mr:.1%}, MissPenalty={mp}ns")
    print(f"  AMAT = {ht} + {mr:.2%} × {mp} = {result:.2f}ns\n")

    # 两级Cache分析
    print("=== 两级Cache的AMAT ===")
    ht_l1 = 1; mr_l1 = 0.05;  # L1
    ht_l2 = 10; mr_l2 = 0.20;  # L2 (L1缺失时访问)
    mp_mem = 100                # 主存访问

    amat_l2 = ht_l2 + mr_l2 * mp_mem
    amat_total = ht_l1 + mr_l1 * amat_l2
    print(f"  L1: HitTime={ht_l1}ns, MissRate={mr_l1:.1%}")
    print(f"  L2: HitTime={ht_l2}ns, MissRate={mr_l2:.1%}")
    print(f"  主存MissPenalty={mp_mem}ns")
    print(f"  AMAT(L2) = {ht_l2} + {mr_l2:.1%} × {mp_mem} = {amat_l2:.1f}ns")
    print(f"  AMAT(总) = {ht_l1} + {mr_l1:.1%} × {amat_l2:.1f} = {amat_total:.2f}ns\n")

    # 因素影响分析
    print("=== 各因素影响方向 ===")
    factors = [
        ("增大Cache容量", "命中率↑", "命中时间↑", "AMAT可能↓"),
        ("增大块大小",   "缺失率↓(先)", "缺失代价↑", "有最优值"),
        ("提高相联度",   "缺失率↓",    "命中时间↑", "有最优值"),
        ("使用LRU",      "缺失率↓",    "硬件复杂度↑", "AMAT↓"),
        ("使用写回法",   "写速度↑",    "一致性复杂",  "AMAT↓"),
    ]
    print(f"{'因素':<15} {'命中率':<14} {'命中时间/代价':<14} {'AMAT':<10}")
    print("-" * 55)
    for f in factors:
        print(f"{f[0]:<15} {f[1]:<14} {f[2]:<14} {f[3]:<10}")

cache_performance_analysis()

# CPI与Cache的关系
print("\n=== Cache对CPI的影响 ===")
base_cpi = 1.0
lw_sw_ratio = 0.35  # 访存指令比例
for miss_rate in [0.01, 0.05, 0.10, 0.20]:
    miss_penalty_cycles = 50  # 缺失代价50周期
    extra = lw_sw_ratio * miss_rate * miss_penalty_cycles
    cpi = base_cpi + extra
    print(f"  MissRate={miss_rate:.0%}: CPI = {base_cpi} + "
          f"{lw_sw_ratio}×{miss_rate:.0%}×{miss_penalty_cycles} = {cpi:.2f}")
```

- 综合题：给定参数计算AMAT
- 分析题：分析为何实际系统选择某种参数配置
