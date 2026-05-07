# 02-隐私保护技术

## 1. 差分隐私（Differential Privacy）

差分隐私是目前最严格、数学定义最完善的隐私保护框架，其核心思想是：向数据或计算结果中注入精心设计的随机噪声，使得攻击者无法判断某一条特定记录是否参与了计算。

### 1.1 形式化定义（ε-差分隐私）

一个随机化算法 $\mathcal{M}$ 满足 $\varepsilon$-差分隐私，当且仅当对于任意两个**相邻数据集** $D$ 和 $D'$（仅相差一条记录），以及所有可能的输出集合 $S$，满足：

$$\Pr[\mathcal{M}(D) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S]$$

- $\varepsilon$（隐私预算）越小，隐私保护越强，但数据效用越低
- 典型取值范围：$\varepsilon \in [0.1, 10]$，$\varepsilon = 1$ 被认为是较强的隐私保护

### 1.2 全局差分隐私 vs 本地差分隐私

**全局差分隐私（Global DP / Central DP）**：

- 一个可信的数据收集方持有所有原始数据
- 在查询结果上添加噪声
- 噪声量较小，数据效用高
- 需要信任中央服务器

**本地差分隐私（Local DP, LDP）**：

- 每个数据所有者在发送数据之前自行添加噪声
- 不需要信任任何第三方
- 噪声量较大，数据效用较低
- 应用场景：Apple 的用户行为统计、Google 的 RAPPOR

### 1.3 Laplace 机制与 Gaussian 机制

**Laplace 机制**：满足纯 $\varepsilon$-差分隐私

$$\mathcal{M}(D) = f(D) + \text{Lap}\left(\frac{\Delta f}{\varepsilon}\right)$$

- $\Delta f$ 为查询函数 $f$ 的全局敏感度（改变一条记录对结果的最大影响）
- Laplace 分布的尺度参数为 $\Delta f / \varepsilon$

**Gaussian 机制**：满足 $(\varepsilon, \delta)$-差分隐私

$$\mathcal{M}(D) = f(D) + \mathcal{N}\left(0, \sigma^2\right), \quad \sigma \geq \frac{\Delta f \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$$

- 引入松弛项 $\delta$（通常取极小值如 $10^{-5}$），允许以极小概率违反隐私保证
- 噪声量为 $O(\Delta f / \varepsilon \cdot \sqrt{\ln(1/\delta)})$

### 1.4 组合定理

当多个差分隐私机制串联使用时，隐私预算会累积：

**基本组合定理**：若 $\mathcal{M}_1$ 满足 $\varepsilon_1$-DP，$\mathcal{M}_2$ 满足 $\varepsilon_2$-DP，则二者组合满足 $(\varepsilon_1 + \varepsilon_2)$-DP。

**高级组合定理**：对于 $k$ 个机制，每个满足 $(\varepsilon, \delta)$-DP，组合后满足 $(\varepsilon', k\delta + \delta')$-DP，其中：

$$\varepsilon' = \varepsilon \sqrt{2k \ln(1/\delta')} + k\varepsilon(e^\varepsilon - 1)$$

高级组合在 $\varepsilon$ 较小时，累积速度远低于线性增长，这为多次查询提供了更紧的隐私预算估计。

### 1.5 隐私预算（Privacy Budget）

隐私预算 $\varepsilon$ 是差分隐私系统中的核心资源：

- 每次查询消耗一部分预算
- 预算耗尽后，系统拒绝回答新的查询
- 需要精心设计预算分配策略（如：给高价值查询分配更多预算）
- 实际系统中，$\varepsilon$ 的选择需要权衡隐私与效用

### 1.6 差分隐私在机器学习中的应用（DP-SGD）

**DP-SGD（Differentially Private Stochastic Gradient Descent）** 是 Abadi 等人在 2016 年提出的差分隐私训练方法：

1. **梯度裁剪（Gradient Clipping）**：将每个样本的梯度范数裁剪至阈值 $C$，控制单个样本对更新的影响
2. **噪声注入**：向聚合梯度添加 Gaussian 噪声

$$\tilde{g}_t = \frac{1}{B}\left(\sum_i \text{clip}(g_i, C) + \mathcal{N}(0, \sigma^2 C^2 I)\right)$$

3. **隐私记账（Privacy Accounting）**：使用矩量法（Moments Accountant）追踪累计隐私消耗

DP-SGD 的挑战：隐私保护与模型精度之间存在权衡，过大的噪声会导致模型收敛缓慢或精度大幅下降。

## 2. 安全多方计算（Secure Multi-Party Computation, MPC）

安全多方计算允许多个参与方在不泄露各自私有输入的前提下，联合计算一个约定的函数。其形式化安全性定义基于"理想世界/现实世界"范式：现实协议的安全性不应弱于一个存在可信第三方的理想模型。

### 2.1 不经意传输（Oblivious Transfer, OT）

**1-out-of-2 OT** 是 MPC 的基础原语：

- 发送方有两个消息 $m_0$ 和 $m_1$
- 接收方有一个选择位 $b \in \{0, 1\}$
- 接收方获得 $m_b$，但不知道 $m_{1-b}$
- 发送方不知道 $b$ 的值

OT 可以用来构造任意的安全多方计算协议。高效的 OT 扩展（OT Extension）可以从少量基础 OT 生成大量 OT 实例。

### 2.2 混淆电路（Yao's Garbled Circuit）

姚期智在 1986 年提出的混淆电路方法：

1. **电路构造**：将目标函数表示为布尔电路
2. **混淆**：电路的每一根线分配两个随机密钥（对应 0 和 1），每个逻辑门用加密表代替
3. **求值**：接收方通过 OT 获得输入线的密钥，逐门解密得到输出
4. **输出解码**：将输出线的密钥映射回实际值

**优化技术**：Free-XOR、Half Gates、Point-and-Permute，大幅降低了通信和计算开销。

### 2.3 秘密共享方案

**Shamir 秘密共享**（Shamir's Secret Sharing）：

- 将秘密 $s$ 编码为一个 $t-1$ 次多项式 $f(x)$ 的常数项，其中 $f(0) = s$
- 给 $n$ 个参与方各分发一个点 $(x_i, f(x_i))$
- 任意 $t$ 个点可以重建多项式（进而恢复 $s$）
- 少于 $t$ 个点得不到关于 $s$ 的任何信息

**加法秘密共享**：将秘密拆分为 $n$ 个随机数的和，各方持有的份额在线性运算中非常高效。

### 2.4 MPC 框架

| 框架 | 特点 |
|------|------|
| **SPDZ** | 基于秘密共享的 MPC 协议族，支持在线/离线两阶段计算 |
| **ABY** | 支持算术秘密共享、布尔秘密共享和 Yao 混淆电路三种范式的混合协议 |
| **MP-SPDZ** | SPDZ 的多协议扩展，支持多种 MPC 后端 |
| **EMP-toolkit** | 高效的混淆电路实现 |

## 3. 可信执行环境（Trusted Execution Environment, TEE）

TEE 是在 CPU 硬件层面创建的隔离安全区域，即使操作系统被攻破，TEE 内部的代码和数据仍然是安全的。

### 3.1 Intel SGX

**Intel Software Guard Extensions（SGX）**：

- 在 CPU 内创建 **Enclave**（飞地），一种硬件隔离的执行环境
- Enclave 内的代码和数据即使在 root 权限或虚拟机管理器级别也无法被访问
- 支持**远程证明（Remote Attestation）**：向远端验证 Enclave 内运行的代码确实是预期版本
- 内存加密：Enclave 数据在离开 CPU 时自动加密

**应用场景**：隐私数据库查询、密钥管理、区块链智能合约隐私保护

### 3.2 ARM TrustZone

- ARM 架构的 TEE 方案，将处理器分为 **Secure World** 和 **Normal World**
- 广泛应用于移动设备和嵌入式系统
- 运行可信操作系统（如 OP-TEE），处理指纹识别、支付等安全敏感操作
- 与 Intel SGX 不同，TrustZone 是系统级隔离而非进程级隔离

### 3.3 TEE 的安全边界与侧信道攻击

TEE 的安全假设是：硬件和 CPU 微码是可信的。但现实中仍存在安全风险：

- **侧信道攻击**：
  - **Spectre/Meltdown**：利用推测执行的微架构缺陷读取 Enclave 内存
  - **缓存侧信道攻击**：Prime+Probe、Flush+Reload 等通过缓存时序推断 Enclave 内部操作
  - **功耗分析与电磁分析**：通过物理测量获取密钥信息
- **Enclave 接口攻击**：通过精心构造的系统调用利用 Enclave 漏洞
- **物理攻击**：冷启动攻击、硬件植入

**缓解措施**：内存加密增强、常量时间编程、Enclave 代码审计、微码补丁

## 4. 同态加密在隐私计算中的应用

同态加密（Homomorphic Encryption, HE）允许在密文上直接执行计算，计算结果解密后等价于在明文上执行相同操作。

### 4.1 分类

- **部分同态加密（PHE）**：支持单一类型运算（如 RSA 支持乘法、Paillier 支持加法）
- **层次同态加密（LHE/LHE）**：支持有限深度的加法和乘法运算（如 BFV、BGV、CKKS）
- **全同态加密（FHE）**：支持任意次加法和乘法运算，通过自举（Bootstrapping）实现

### 4.2 在隐私计算中的应用

- **纵向联邦学习**：各方加密本地计算的梯度分量，服务器在密文上聚合
- **隐私推理**：将模型部署在云端，用户加密输入数据，云端在密文上执行推理
- **隐私集合求交（PSI）**：同态加密可以构造高效的 PSI 协议

**挑战**：同态加密的计算开销极大（比明文操作慢 $10^3 \sim 10^6$ 倍），通信膨胀显著（密文远大于明文）。CKKS 方案在机器学习推理中较为流行，因为支持近似算术运算。

## 5. 隐私计算平台

### 5.1 蚂蚁摩斯（MORSE）

- 蚂蚁集团开发的隐私计算平台
- 集成 MPC、联邦学习、TEE 等多种技术
- 支持多方安全对齐、安全联合建模、安全联合预测
- 在金融风控、保险定价等场景中大规模落地

### 5.2 华控清交（PrivPy）

- 清华大学背景的隐私计算公司
- 提供多方安全计算平台，支持 Python 编程接口
- 在医疗、政务数据开放等领域有实际应用
- 支持 SQL 风格的隐私查询

### 5.3 其他平台

- **PySyft（OpenMined）**：开源隐私计算框架，社区活跃
- **SecretFlow（蚂蚁）**：开源隐私计算框架，支持多种计算范式
- **联邦学习 + TEE 融合**：越来越多平台将多种隐私技术组合使用，形成纵深防御

## 参考资料

- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014)
- Abadi et al., "Deep Learning with Differential Privacy" (2016)
- Yao, "How to Generate and Exchange Secrets" (1986)
- Shamir, "How to Share a Secret" (1979)
- Intel SGX 官方文档
- 《隐私计算：原理、算法与应用》，机械工业出版社
