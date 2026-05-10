# SPIN与NuSMV


# SPIN与NuSMV模型检验工具

一、SPIN模型检验器

## 一、SPIN模型检验器


SPIN是贝尔实验室开发的显式状态模型检验器，专门用于验证并发系统的正确性。它使用Promela语言建模，通过LTL性质表达验证需求。


### 1.1 SPIN特性


| 特性 | 说明 |
| --- | --- |
| 建模语言 | Promela (Process Meta Language) |
| 验证逻辑 | LTL (线性时序逻辑) |
| 搜索算法 | 深度优先搜索 (DFS)、广度优先搜索 (BFS) |
| 状态存储 | 显式枚举所有可达状态 |
| 优化技术 | 偏序约简 (Partial Order Reduction)、状态压缩 |
| 反例生成 | 自动生成违反性质的反例执行序列 |


### 1.2 Promela语言基础


> **Note:** **Promela基本元素：**
>
>
>
>
> /* 变量声明 */
>
>
> bool flag;           /* 布尔变量 */
>
>
> byte count = 0;      /* 字节变量 (0-255) */
>
>
> int x;               /* 整型变量 */
>
>
> chan ch = [8] of {byte};  /* 容量为8的通道 */
>
>
>
>
> /* 进程定义 */
>
>
> proctype P(byte id) {
>
>
> do
>
>
> :: (count < 10) -> count = count + 1;
>
>
> :: else -> break;
>
>
> od;
>
>
> }
>
>
>
>
> /* 启动进程 */
>
>
> init {
>
>
> run P(1);
>
>
> run P(2);
>
>
> }


### 1.3 经典示例：互斥锁验证


> **Note:** **Promela互斥锁模型：**
>
>
>
>
> bool lock = false;  /* 互斥锁 */
>
>
> byte ncrit = 0;     /* 临界区进程计数 */
>
>
>
>
> proctype Process(byte id) {
>
>
> do
>
>
> :: true ->
>
>
> /* 非临界区 */
>
>
> skip;
>
>
> /* 请求锁 */
>
>
> atomic { !lock -> lock = true };
>
>
> /* 临界区 */
>
>
> ncrit = ncrit + 1;
>
>
> assert(ncrit == 1);  /* 互斥性 */
>
>
> ncrit = ncrit - 1;
>
>
> /* 释放锁 */
>
>
> lock = false;
>
>
> od;
>
>
> }
>
>
>
>
> init {
>
>
> run Process(1);
>
>
> run Process(2);
>
>
> }


### 1.4 LTL性质表达


| LTL算子 | 含义 | 示例 |
| --- | --- | --- |
| [] p (G p) | 全局性质：p在所有时刻为真 | [] (ncrit <= 1) 安全性 |
| <> p (F p) | 最终性质：p在某时刻为真 | <> (count == 10) 活性 |
| X p | 下一时刻：p在下一状态为真 | X (state == READY) |
| p U q | 直到：p一直为真直到q为真 | (waiting) U (granted) |
| -> | 蕴含：p -> q 等价于 !p \|\| q | [] (request -> <> response) |

二、NuSMV

## 二、NuSMV模型检验器


NuSMV是基于符号模型检验（BDD/SAT）的模型检验器，适合验证有限状态系统的CTL和LTL性质。


### 2.1 NuSMV特性


| 特性 | 说明 |
| --- | --- |
| 建模语言 | NuSMV语言（基于状态机） |
| 验证逻辑 | CTL + LTL + PSL |
| 搜索算法 | 符号模型检验 (BDD)、有界模型检验 (SAT) |
| 状态表示 | 隐式（BDD编码，可处理更大状态空间） |
| 同步/异步 | 支持同步和异步进程组合 |


### 2.2 NuSMV模型示例


> **Note:** **NuSMV简单状态机模型：**
>
>
>
>
> MODULE main
>
>
> VAR
>
>
> state : {idle, running, stopped};
>
>
> button : boolean;
>
>
> speed : 0..100;
>
>
>
>
> ASSIGN
>
>
> init(state) := idle;
>
>
> init(speed) := 0;
>
>
>
>
> next(state) :=
>
>
> case
>
>
> state = idle & button : running;
>
>
> state = running & button : stopped;
>
>
> state = stopped & button : idle;
>
>
> TRUE : state;
>
>
> esac;
>
>
>
>
> next(speed) :=
>
>
> case
>
>
> state = running & speed < 100 : speed + 10;
>
>
> state = stopped : 0;
>
>
> TRUE : speed;
>
>
> esac;
>
>
>
>
> LTLSPEC
>
>
> G (state = running -> F state = stopped)
>
>
>
>
> CTLSPEC
>
>
> AG (state = running -> AF state = stopped)
>
>
> AG (speed <= 100)


### 2.3 CTL vs LTL


| 特性 | LTL (线性时序逻辑) | CTL (计算树逻辑) |
| --- | --- | --- |
| 路径量化 | 隐式（所有路径） | 显式（A=所有路径, E=存在路径） |
| 表达力 | 部分可比较 | 部分可比较 |
| 工具 | SPIN | NuSMV, CTL模型检验器 |
| 典型公式 | G(p -> Fq) | AG(p -> AFq) |

三、SPIN vs NuSMV

## 三、SPIN vs NuSMV 对比


| 维度 | SPIN | NuSMV |
| --- | --- | --- |
| 状态空间表示 | 显式枚举 | 符号化 (BDD/SAT) |
| 擅长领域 | 并发协议、通信系统 | 硬件设计、控制系统 |
| 建模风格 | 进程通信（通道） | 状态机（变量赋值） |
| 验证逻辑 | LTL + 断言 | CTL + LTL + PSL |
| 状态爆炸 | 偏序约简 | BDD符号化、SAT BMC |
| 上手难度 | 较易 | 中等 |
| 工业应用 | 通信协议验证 | 硬件、安全关键系统 |

四、工业应用

## 四、工业应用案例


### 4.1 SPIN工业应用


- **通信协议：**
   验证TCP/IP协议栈、蓝牙协议、路由协议的正确性
- **操作系统：**
   验证调度算法、死锁自由性、内存管理
- **分布式系统：**
   验证共识协议（Paxos、Raft）的安全性
- **NASA：**
   验证航天器控制软件的并发行为


### 4.2 NuSMV工业应用


- **硬件设计：**
   验证芯片控制逻辑、FPGA设计
- **安全关键系统：**
   铁路信号、航空电子系统
- **安全协议：**
   验证TLS握手、Kerberos协议
- **汽车ECU：**
   验证嵌入式控制逻辑

**模型检验的挑战——状态爆炸：**
系统状态数随变量数指数增长。应对策略：SPIN使用偏序约简和状态压缩；NuSMV使用BDD符号化表示和SAT有界模型检验(BMC)。抽象（abstraction）是应对状态爆炸的通用策略。
========================================
  文件总结
========================================
  主题：SPIN与NuSMV模型检验工具
  内容概要：
    1. SPIN - 显式状态模型检验器，Promela语言，LTL验证，并发系统
    2. NuSMV - 符号模型检验器，BDD/SAT，CTL+LTL，硬件/控制系统
    3. Promela语言基础 - 进程/通道/选择/循环
    4. LTL算子 - G(全局)/F(最终)/X(下一)/U(直到)
    5. 工业应用 - SPIN验证通信协议，NuSMV验证硬件/安全关键系统
  重点知识：
    - SPIN偏序约简 vs NuSMV BDD符号化应对状态爆炸
    - LTL vs CTL的区别（线性 vs 分支）
    - 模型检验自动产生反例，这是相比测试的主要优势
========================================


<!-- Converted from: 02_SPIN与NuSMV.html -->
