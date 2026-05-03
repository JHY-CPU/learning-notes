# 08_分组卷积 (Group Conv) 与 ShuffleNet

## 核心概念

- **分组卷积定义**：将输入通道分成 $G$ 组，每组独立进行卷积，输出也是各组输出的拼接。分组数 $G$ 越大，参数量和计算量越小。
- **计算效率**：分组卷积的计算量为标准卷积的 $1/G$。参数量为 $C_{in}/G \times C_{out}/G \times K^2 \times G = C_{in} \times C_{out} \times K^2 / G$。
- **通道间信息隔离问题**：分组卷积的每组只在本组内进行信息融合，不同组之间的信息没有交流，这限制了模型的表达能力。
- **ShuffleNet的通道混洗（Channel Shuffle）**：通过将分组卷积后的特征图进行通道重排，使不同组之间的信息能够交互，解决了分组卷积的信息隔离问题。
- **AlexNet的历史贡献**：分组卷积最初由AlexNet引入（受限于GTX 580的1.5GB显存，将网络拆到两块GPU上训练），后来被发现可以提升模型精度。
- **ResNeXt的基数（Cardinality）**：ResNeXt将分组卷积作为核心构建块，验证了增加分组数（基数）比增加深度或宽度更有效地提升模型性能。

## 数学推导

**标准卷积 vs 分组卷积参数量：**
$$
\text{Param}_{std} = C_{in} \times C_{out} \times K^2 + C_{out}
$$
$$
\text{Param}_{group} = \frac{C_{in}}{G} \times \frac{C_{out}}{G} \times K^2 \times G + C_{out} = \frac{C_{in} \times C_{out} \times K^2}{G} + C_{out}
$$

**ShuffleNet中的通道混洗操作：**
假设输入特征图有 $G \times n$ 个通道（$G$ 组，每组 $n$ 个通道），通道混洗的步骤为：
- 将通道维度reshape为 $(G, n)$
- 转置为 $(n, G)$
- flatten回 $G \times n$ 维

数学上，通道混洗可表示为：
$$
\text{Shuffle}(x) = \text{reshape}^{-1}(\text{transpose}(\text{reshape}(x, (G, n)), 0, 1))
$$

**计算量比较（ResNeXt vs ResNet）：**
在相近参数量下，ResNeXt通过增加基数 $G$（如 $G=32$）使每个组的宽度变窄，总Flops与ResNet相近但精度更高。

## 直观理解

分组卷积就像把一个大型团队拆分成多个小组，每个小组独立处理不同的任务。但问题在于，如果小组之间从不交流，每个小组只了解信息的子集，容易产生"部门墙"。ShuffleNet的通道混洗就像定期的人员轮换——把不同小组的成员重新分配，确保信息在全团队流通。这既保留了分组卷积的高效性（小组内工作），又解决了信息隔离问题（通过混洗实现跨组交流）。

## 代码示例

```python
import torch
import torch.nn as nn

# 标准卷积 vs 分组卷积参数量对比
in_c, out_c, k = 128, 128, 3
std_conv = nn.Conv2d(in_c, out_c, k, padding=k//2, groups=1)  # 标准卷积
group_conv_32 = nn.Conv2d(in_c, out_c, k, padding=k//2, groups=32)  # 32组

std_params = sum(p.numel() for p in std_conv.parameters())
group_params = sum(p.numel() for p in group_conv_32.parameters())
print(f"标准卷积参数量: {std_params}")
print(f"32组分组卷积参数量: {group_params}")
print(f"参数量比: {group_params / std_params:.3f}")

# 通道混洗实现
def channel_shuffle(x, groups):
    batch_size, channels, h, w = x.shape
    assert channels % groups == 0
    channels_per_group = channels // groups
    # reshape -> (batch, groups, channels_per_group, h, w)
    x = x.view(batch_size, groups, channels_per_group, h, w)
    # transpose -> (batch, channels_per_group, groups, h, w)
    x = x.transpose(1, 2).contiguous()
    # flatten -> (batch, channels, h, w)
    x = x.view(batch_size, -1, h, w)
    return x

# 测试通道混洗
x = torch.arange(24).view(1, 6, 2, 2).float()
print("混洗前通道:\n", x[0, :, 0, 0])
shuffled = channel_shuffle(x, groups=3)
print("混洗后通道:\n", shuffled[0, :, 0, 0])
```

## 深度学习关联

- **ShuffleNet V1/V2**：ShuffleNet V1将通道混洗与分组卷积结合，在移动端设备上达到优于MobileNet的精度-速度平衡；ShuffleNet V2进一步提出高效网络设计的4条实用准则（如输入输出通道数相等、分组数适中）。
- **ResNeXt的基数创新**：ResNeXt通过实验证明，在计算量固定的条件下，增加基数（分组数）比加深网络或增加宽度能获得更高的精度提升，这一发现影响了后续EfficientNet等模型的设计。
- **Transformer中的多头注意力**：多头自注意力机制中的每个头可以看作是"特征维度上的分组"，每个头独立计算注意力后拼接，类似于分组卷积的思想。通道混洗也在某些Transformer变体中被用于增强头间信息交流。
