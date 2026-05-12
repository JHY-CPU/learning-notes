# 23_DMA方式概述

## 核心概念

- **DMA(Direct Memory Access，直接存储器访问)**：IO设备与主存之间直接进行数据传送，不需要CPU干预
- **核心思想**：绕过CPU，让IO设备直接读写主存
- **意义**：解放CPU，高速批量数据传输成为可能

### DMA方式的基本特征

| 特征 | 说明 |
|------|------|
| 数据传送路径 | IO设备 ↔ 主存(不经CPU) |
| CPU参与度 | 仅预处理和后处理，传送过程不参与 |
| 数据传送单位 | 可以是单个字，也可以是数据块(成组) |
| CPU利用率 | 最高，几乎不影响CPU工作 |
| 硬件复杂度 | 高，需要独立的DMA控制器 |
| 适用场景 | 高速批量数据传输(磁盘、网卡等) |

### DMA vs 程序中断 vs 程序查询

| 对比项 | 程序查询 | 程序中断 | DMA |
|--------|----------|----------|-----|
| 数据路径 | 设备→CPU→主存 | 设备→CPU→主存 | 设备→主存(直接) |
| CPU每字开销 | 查询+传送 | 中断响应+传送 | 仅预/后处理 |
| 每次传送 | 1个字 | 1个字 | 一个数据块 |
| 适合速度 | 低速 | 低速~中速 | 高速 |

## 原理分析

### DMA的基本原理

```
               数据总线
          ┌───────┤───────┐
          │               │
     ┌────┴────┐    ┌────┴────┐
     │  主存    │    │   CPU   │
     └────┬────┘    └─────────┘
          │
     ┌────┴────┐
     │  DMA    │  ← DMA控制器(独立于CPU)
     │  控制器  │
     └────┬────┘
          │
     IO设备(如磁盘)
```

**关键点**：
- DMA控制器可以独立控制总线，直接访问主存
- 数据传送期间CPU不需要参与
- CPU和DMA可能竞争访问主存，需要总线仲裁

### DMA传送的三个阶段

```
阶段1: 预处理(软件，CPU执行)
  CPU设置DMA控制器参数：
  - 传送方向(读/写)
  - 主存起始地址
  - 传送字数(计数器初值)
  - 设备地址
  ↓
阶段2: 数据传送(硬件，DMA控制器执行)
  DMA控制器逐个(或成组)传送数据
  每传一个字：地址+1，计数器-1
  计数器到0 → 传送结束
  ↓
阶段3: 后处理(中断，CPU执行)
  DMA发出中断请求
  CPU响应中断
  检查传送状态，处理错误
  结束IO操作
```

### DMA的工作优势

**以读取磁盘1KB数据为例**：

**程序中断方式**：
- 需要1024次中断
- 每次中断开销20μs
- 总中断开销 = $1024 \times 20 = 20480\mu s$
- CPU用于IO的开销 = 20480μs

**DMA方式**：
- 预处理：10μs
- 数据传送：DMA控制器自动完成，CPU不参与
- 后处理(1次中断)：20μs
- CPU总开销 = 30μs
- **CPU开销减少到原来的0.15%**

## 直观理解

**DMA类比**：搬家
- **程序查询/中断**：你自己一趟一趟搬(CPU参与每个字节)
- **DMA**：请搬家公司，你只管吩咐(预处理)和验收(后处理)，搬家公司自己搬(DMA控制器)
- 搬家过程中你可以做其他事情(CPU执行其他程序)

## 代码/模拟

### Python模拟DMA传送过程

```python
"""DMA传送过程模拟 - 适用于408考研复习"""

class DMAController:
    """DMA控制器模拟"""

    def __init__(self):
        self.MAR = 0       # 主存地址寄存器
        self.DAR = 0       # 设备地址寄存器
        self.WC = 0        # 字计数器
        self.data_buffer = 0

    def preload(self, mem_addr, device_addr, word_count):
        """预处理: CPU设置DMA控制器参数"""
        self.MAR = mem_addr
        self.DAR = device_addr
        self.WC = word_count
        print(f"【预处理 - CPU设置DMA参数】")
        print(f"  MAR(主存起始地址) ← {mem_addr:#06x}")
        print(f"  DAR(设备地址)    ← {device_addr:#06x}")
        print(f"  WC(传送字数)     ← {word_count}")

    def transfer_block(self, memory, device_data):
        """数据传送阶段 - DMA控制器自动完成"""
        print(f"\n【数据传送 - DMA自动完成, 不需CPU干预】")
        transferred = 0
        for i in range(self.WC):
            # 模拟: 设备 → 主存
            data = device_data[i] if i < len(device_data) else 0
            memory[self.MAR + i] = data
            transferred += 1
            print(f"  传送{transferred}: 设备[{self.DAR + i}] → "
                  f"主存[{self.MAR + i:#06x}] = {data:#04x}")
            self.WC -= 1

        print(f"\n  传送完成! 共传送{transferred}个字")
        print(f"  WC = 0, 发出中断请求通知CPU")

    def post_process(self):
        """后处理: CPU响应DMA中断, 做善后工作"""
        print(f"\n【后处理 - CPU响应DMA中断】")
        print(f"  检查传送是否正确")
        print(f"  更新程序中的缓冲区指针")
        print(f"  释放DMA控制器")

# 完整DMA传送过程
print("=" * 50)
print("DMA传送完整过程模拟")
print("=" * 50)

dma = DMAController()
memory = [0] * 0x1000  # 简化主存
disk_data = [0x41, 0x42, 0x43, 0x44, 0x45]  # 磁盘读出的数据

# Step 1: 预处理 (CPU参与)
dma.preload(mem_addr=0x100, device_addr=0x00, word_count=5)

# Step 2: 数据传送 (DMA自主完成, CPU可执行其他程序)
dma.transfer_block(memory, disk_data)

# Step 3: 后处理 (CPU参与)
dma.post_process()

# 三种I/O方式对比
print("\n" + "=" * 50)
print("三种I/O方式CPU开销对比 (传送1000个字)")
print("=" * 50)
print(f"{'方式':<12} {'每字CPU开销':<15} {'总CPU开销':<15}")
print("-" * 45)
print(f"{'程序查询':<12} {'查询+传送':<15} {'1000 × 20μs = 20ms':<15}")
print(f"{'程序中断':<12} {'中断响应+传送':<15} {'1000 × 10μs = 10ms':<15}")
print(f"{'DMA':<12} {'仅预/后处理':<15} {'30μs (几乎为0!)':<15}")
```

## 知识关联

### 跨章节联系

| 相关知识点 | 联系 |
|------------|------|
| **总线仲裁(第6章)** | DMA与CPU竞争总线，需要仲裁机制 |
| **存储器(第5章)** | DMA直接访问主存，涉及地址和数据通路 |
| **Cache一致性(第5章)** | DMA绕过Cache修改主存，可能导致Cache不一致 |

### 易错陷阱

1. **DMA不等于零CPU开销**：预处理和后处理仍需CPU参与
2. **DMA控制器不是CPU**：没有指令系统，不能执行程序
3. **DMA仍需总线**：DMA传送仍然占用总线带宽
4. **DMA优先级通常高于中断**：因为DMA有时间限制(如磁盘旋转窗口)
