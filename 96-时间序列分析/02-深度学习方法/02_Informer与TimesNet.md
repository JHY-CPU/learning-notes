# Informer与TimesNet


## 一、长序列时间预测（LTSF）的挑战


传统Transformer直接应用于长时间序列预测面临以下核心挑战：


| 挑战 | 描述 | 复杂度 |
| --- | --- | --- |
| 自注意力的二次复杂度 | 标准注意力矩阵为 O(L²) | O(L²d) |
| 内存瓶颈 | 存储注意力矩阵消耗大量显存 | O(L²) |
| 解码效率 | 逐token生成预测，推理慢 | O(L) |
| 长程依赖稀释 | 注意力权重分散，关键信息被稀释 | — |


> **Important:** **核心争论：**
> Zeng et al. (2023) 在 "Are Transformers Effective for Time Series Forecasting?" 中指出，简单的线性模型（DLinear）在很多数据集上甚至优于复杂的Transformer模型，引发了"Transformer是否真正适合时间序列"的讨论。


## 二、Informer（AAAI 2021 最佳论文）


Informer由Zhou et al.提出，是首个专门针对长序列时间预测的Transformer变体，获AAAI 2021最佳论文奖。


### 2.1 ProbSparse 自注意力


标准自注意力中，大部分查询的注意力分布是稀疏的——只有少数查询需要关注所有键。ProbSparse 利用这一特性，只保留"信息量最大"的查询。


$$
KL散度度量查询的信息量：
                M(qi, K) = ln ∑j eqikjT/√d - (1/LK) ∑j qikjT/√d
                选择 Top-u 个 M 值最大的查询，复杂度从 O(L²) 降为 O(L ln L)
$$


### 2.2 蒸馏操作


为了进一步减少层数间的维度，Informer引入了蒸馏（Distilling）操作，将特征维度减半：


$$
Xj+1d = MaxPool(ELU(Conv1d(Xj)))
                其中 Conv1d 使用 kernel_size=3, stride=2
$$


### 2.3 生成式解码


不同于逐token自回归生成，Informer一次性生成整个预测序列：将预测部分的token填充为零，一次性送入decoder，直接输出所有预测值。


## 三、Autoformer 与 PatchTST


### 3.1 Autoformer（NeurIPS 2021）


Wu et al. 提出用**自相关（Auto-Correlation）**替代自注意力。自相关基于频域分析，发现序列间的周期性依赖关系。


$$
自相关计算：
                Rxx(τ) = ∑ Xt · Xt+τ / T
                选择 Top-k 个周期对应的时延 τ，进行时延聚合
$$


- 复杂度：O(L log L)（利用FFT计算自相关）
- 优势：天然捕捉周期性模式，适合具有明显季节性的时间序列


### 3.2 PatchTST（ICLR 2023）


Nie et al. 提出将时间序列切分为不重叠的**Patch**（子序列），每个Patch作为一个token输入Transformer。


- Patch大小 P 和步长 S 是关键超参数
- 大幅减少token数量：L=512, P=16 → 32个token
- 保留了局部语义信息
- 支持通道独立（Channel Independence）：每个变量独立建模


## 四、TimesNet（ICLR 2023）


Wu et al. 提出的核心思想：将1D时间序列转换为2D张量，用2D卷积提取**多周期内的变化模式**。


### 4.1 1D → 2D 转换


$$
假设检测到的前 K 个周期为 {p1, p2, ..., pK}
                对于周期 pi：将长度为 L 的序列 reshape 为 (L//pi, pi) 的2D张量
                行=周期内的位置，列=第几个周期
$$


### 4.2 TimesBlock


1. **周期检测：**
   用FFT对1D序列做频谱分析，找到Top-K个频率/周期
2. **2D转换：**
   对每个周期，将1D序列reshape为2D
3. **2D卷积：**
   用参数高效的Inception Block提取2D模式
4. **自适应聚合：**
   对K个周期的结果做加权融合


> **Note:** **TimesNet的巧妙之处：**
> 时间序列的复杂变化可以分解为多个周期性模式的叠加。2D卷积能同时捕捉"周期内变化"和"跨周期变化"，比1D方法更全面。


## 五、时序大模型简介


2023-2024年，大语言模型（LLM）也被应用于时间序列预测，形成新的研究方向。


| 模型 | 核心思想 | 发表 |
| --- | --- | --- |
| **LLMTime** | 直接将时间序列作为token输入GPT/LLaMA | NeurIPS 2023 |
| **Time-LLM** | 用重编程层将时序数据对齐到LLM | ICLR 2024 |
| **GPT4TS** | 冻结GPT-2的注意力层，微调其他层 | NeurIPS 2023 |
| **Lag-Llama** | 基于LLaMA架构的时序基础模型 | 2024 |
| **MOMENT** | 基于T5的通用时序基础模型 | ICML 2024 |


> **Important:** **注意事项：**
> 时序大模型目前仍处于探索阶段。在很多基准测试上，精心调参的经典模型（如PatchTST、DLinear）仍然具有竞争力。大模型的优势主要体现在零样本预测和跨数据集泛化上。


## 六、模型综合对比


| 模型 | 年份 | 核心机制 | 复杂度 | 优势 | 不足 |
| --- | --- | --- | --- | --- | --- |
| Informer | 2021 | ProbSparse注意力 | O(L log L) | 长序列处理 | 实现复杂 |
| Autoformer | 2021 | 自相关 | O(L log L) | 周期性捕捉 | 非周期数据弱 |
| PatchTST | 2023 | Patch+标准注意力 | O((L/P)²) | 简单高效 | Patch选择敏感 |
| TimesNet | 2023 | 1D转2D+2D卷积 | O(KL) | 多周期建模 | 周期检测依赖FFT |
| DLinear | 2023 | 趋势+季节分解+线性 | O(L) | 极简高效 | 表达能力有限 |


## 七、Python 实战：使用 TimesNet


> **Example:** ### 示例：使用 Time-Series-Library 框架
>
>
> ```
> # Time-Series-Library 安装：pip install git+https://github.com/thuml/Time-Series-Library.git
>
> # 方法一：使用官方库运行TimesNet
> # 终端命令（ETTh1数据集，预测96步）：
> # python -u run.py \
> #   --task_name long_term_forecast \
> #   --is_training 1 \
> #   --root_path ./dataset/ETT-small/ \
> #   --data_path ETTh1.csv \
> #   --model_id ETTh1_96_96 \
> #   --model TimesNet \
> #   --data ETTh1 \
> #   --features M \
> #   --seq_len 96 \
> #   --label_len 48 \
> #   --pred_len 96 \
> #   --e_layers 2 \
> #   --d_layers 1 \
> #   --factor 3 \
> #   --enc_in 7 \
> #   --dec_in 7 \
> #   --c_out 7 \
> #   --d_model 16 \
> #   --d_ff 32 \
> #   --top_k 5 \
> #   --des 'Exp' \
> #   --itr 1
>
> # 方法二：手动实现核心的2D转换逻辑
> import torch
> import torch.nn as nn
> import numpy as np
>
> def FFT_for_Period(x, k=2):
>     """用FFT检测Top-K个周期"""
>     # x: (B, T, C)
>     xf = torch.fft.rfft(x, dim=1)
>     frequency_list = abs(xf).mean(dim=0).mean(dim=-1)
>     frequency_list[0] = 0  # 去掉直流分量
>     top_k = torch.topk(frequency_list, k)
>     top_k_indices = top_k.indices.detach().cpu().numpy()
>     period = x.shape[1] // top_k_indices  # 周期长度
>     return period, abs(xf).mean(dim=-1)[:, top_k_indices]
>
> def time_series_to_2d(x, period):
>     """将1D序列转为2D张量"""
>     B, T, C = x.shape
>     if T % period != 0:
>         padding = period - T % period
>         x = torch.cat([x, x[:, -padding:, :]], dim=1)
>         T = x.shape[1]
>     x_2d = x.reshape(B, T // period, period, C)  # (B, rows, cols, C)
>     return x_2d
>
> # 测试
> B, T, C = 32, 96, 7
> x = torch.randn(B, T, C)
> periods, weights = FFT_for_Period(x, k=3)
> print(f"检测到的周期: {periods}")
>
> for p in periods:
>     x_2d = time_series_to_2d(x, p)
>     print(f"周期={p}: 1D shape {x.shape} → 2D shape {x_2d.shape}")
> ```


## 总结


- Informer用ProbSparse注意力将复杂度从O(L²)降到O(L log L)
- Autoformer用自相关替代注意力，利用FFT捕捉周期性
- PatchTST将序列分Patch作为token，简单且效果优秀
- TimesNet将1D转2D用2D卷积，同时捕捉多周期内和跨周期变化
- DLinear的出现引发了"Transformer是否真的适合时序"的讨论
- 时序大模型（LLMTime、Time-LLM等）是新方向，但尚未完全成熟
- 选择模型时应根据数据特点和计算资源综合考虑


<!-- Converted from: 02_Informer与TimesNet.html -->
