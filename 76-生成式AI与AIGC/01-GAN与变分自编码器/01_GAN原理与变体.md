# GAN原理与变体

*从原始GAN到StyleGAN — 生成对抗网络的演进*


### 1.1 核心思想


**生成对抗网络（Generative Adversarial Network, GAN）**由Ian Goodfellow于2014年提出，包含两个相互博弈的网络：


- **生成器（Generator, G）**
   ：接收随机噪声z，生成假样本G(z)，目标是"欺骗"判别器
- **判别器（Discriminator, D）**
   ：判断输入是真实样本还是生成样本，目标是"识破"生成器


两者通过**极小极大博弈（Minimax Game）**交替训练：


$$
min_G max_D V(D,G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]
$$


博弈的纳什均衡点：生成器的分布 p_g 等于真实数据分布 p_data，判别器对任何输入输出0.5。


### 1.2 训练过程


```
# GAN训练伪代码
for epoch in range(epochs):
    # === 训练判别器 ===
    for k steps:
        # 采样真实数据 batch x
        # 采样噪声 z
        # 更新D: max (log D(x) + log(1-D(G(z))))
        # 梯度上升

    # === 训练生成器 ===
    # 采样噪声 z
    # 更新G: min log(1-D(G(z)))  等价于 max log(D(G(z)))
    # 梯度下降

# 实际训练中，G通常使用 max log(D(G(z))) 替代 min log(1-D(G(z)))
# 原因：训练早期G很弱，log(1-D(G(z)))接近0，梯度极小
# 使用 log(D(G(z))) 在训练早期提供更大的梯度
```


### 2.1 模式崩塌（Mode Collapse）


生成器只学会生成少数几种样本，无法覆盖真实数据的全部模式：


- 生成器找到了判别器的"盲点"，反复生成骗过判别器的同一种样本
- 表现：生成的图像多样性极低（如只生成一种数字）
- 缓解方法：mini-batch discrimination、unrolled GAN、多个判别器


### 2.2 训练不稳定


- **梯度消失**
   ：当D训练得太好时，G的梯度趋近于0，无法学习
- **振荡**
   ：G和D的训练速度不匹配，导致反复震荡不收敛
- **评估困难**
   ：没有明确的损失函数指示生成质量


### 2.3 评估指标


| 指标 | 说明 | 衡量维度 |
| --- | --- | --- |
| Inception Score (IS) | 用Inception网络评估生成图像的质量和多样性 | 质量+多样性 |
| Fréchet Inception Distance (FID) | 真实和生成图像在Inception特征空间的距离 | 越低越好 |
| Precision & Recall | 分别衡量生成质量和多样性 | 质量、多样性分离 |


$$
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
$$


### 3.1 动机


原始GAN的JS散度在两个分布不重叠时为常数（梯度为0），导致训练不稳定。WGAN使用**Wasserstein距离（Earth Mover's Distance）**替代JS散度。


$$
W(p_data, p_g) = inf_{γ∈Π(p_data, p_g)} E_{(x,y)~γ}[||x - y||]
$$


等价于求解：


$$
W(p_data, p_g) = sup_{||f||_L ≤ 1} [E_x~p_data[f(x)] - E_x~p_g[f(x)]]
$$


### 3.2 关键改进


- 判别器改为
   **Critic**
   （不输出概率，输出实数分数）
- 去掉D输出层的Sigmoid
- 损失函数不取log
- **权重裁剪**
   （WGAN）或
   **梯度惩罚**
   （WGAN-GP）满足Lipschitz约束


```
# WGAN损失函数
# Critic损失: L_D = D(fake) - D(real)  （梯度上升）
# 生成器损失: L_G = -D(fake)           （梯度下降）

# WGAN-GP梯度惩罚
L_GP = λ · E[(||∇_x̂ D(x̂)||₂ - 1)²]
# x̂ = ε·x_real + (1-ε)·x_fake,  ε ~ U(0,1)
# 强制Critic的梯度范数接近1（1-Lipschitz）
```


> **Note:** **WGAN的训练稳定性：**
> Wasserstein距离处处连续可微，即使两个分布不重叠也能提供有意义的梯度。WGAN-GP进一步提高了训练稳定性，是后续GAN变体的常用判别器训练策略。


### 4.1 架构设计准则


DCGAN（Deep Convolutional GAN, 2016）将CNN引入GAN，提出了几条重要设计准则：


- 用
   **步长卷积**
   替代池化层（D中用stride=2的卷积下采样）
- G中用
   **转置卷积**
   上采样
- 使用
   **Batch Normalization**
   （G和D中都使用，但D的输入层和G的输出层除外）
- G使用
   **ReLU**
   激活（输出层用Tanh）
- D使用
   **LeakyReLU**
   激活


```
# DCGAN生成器结构
z (100维噪声) → FC(4×4×1024) → Reshape
→ TransposedConv(512, 4×4, stride=2) + BN + ReLU   → 8×8×512
→ TransposedConv(256, 4×4, stride=2) + BN + ReLU   → 16×16×256
→ TransposedConv(128, 4×4, stride=2) + BN + ReLU   → 32×32×128
→ TransposedConv(3, 4×4, stride=2) + Tanh          → 64×64×3
```


### 5.1 cGAN


条件GAN在G和D中都加入条件信息c（如类别标签、文本描述等），实现可控生成：


$$
min_G max_D V(D,G) = E[log D(x|c)] + E[log(1 - D(G(z|c)|c))]
$$


- G: 输入噪声z和条件c，生成与条件匹配的样本
- D: 输入样本和条件c，判断样本是否真实且与条件匹配


### 5.2 Pix2Pix


成对图像翻译框架，用**配对数据**学习图像到图像的映射：


- **G结构**
   ：U-Net（带跳跃连接保留细节）
- **D结构**
   ：PatchGAN（判别器输出N×N矩阵，每个值判断图像的一个局部区域）
- **损失函数**
   ：L = L_GAN + λ·L_L1（L1损失保证像素级匹配）
- 应用：线稿→彩色图、卫星图→地图、白天→夜晚


### 5.3 CycleGAN


不需要配对数据的图像翻译，通过**循环一致性损失**保证翻译质量：


```
# CycleGAN: 域A ↔ 域B
# 两个生成器: G: A→B, F: B→A
# 两个判别器: D_A（判别域A）, D_B（判别域B）

# 循环一致性损失:
# A → G(A) → F(G(A)) ≈ A（内容保留）
# B → F(B) → G(F(B)) ≈ B

L_cycle = E[||F(G(A)) - A||₁] + E[||G(F(B)) - B||₁]

# 总损失:
L = L_GAN(G, D_B, A, B) + L_GAN(F, D_A, B, A) + λ·L_cycle
```


应用：马↔斑马、照片↔油画、夏天↔冬天


### 6.1 StyleGAN v1（2019）


NVIDIA提出，实现了前所未有的高分辨率、高保真面部生成：


- **映射网络（Mapping Network）**
   ：8层MLP将噪声z映射到中间潜空间W（解耦更好）
- **自适应实例归一化（AdaIN）**
   ：在每个卷积层注入风格信息
- **噪声注入**
   ：每层添加随机噪声控制细节变化（发丝、毛孔等）
- **样式混合（Style Mixing）**
   ：训练时随机切换不同层的风格


```
# StyleGAN生成器结构
z → Mapping Network → w (中间潜码)
w → AdaIN(每个卷积层) → 控制风格
Random Noise → 每层注入 → 控制随机细节

# AdaIN操作
AdaIN(x_i, y) = y_s,i · (x_i - μ(x_i)) / σ(x_i) + y_b,i
# y_s和y_b由w通过学习的仿射变换得到
```


### 6.2 StyleGAN v2（2020）


- **权重调制/解调制**
   ：替代AdaIN，消除水滴状伪影
- **路径正则化**
   ：鼓励W空间更平滑
- **渐进式训练改进**
   ：用残差连接替代双线性上/下采样
- 生成1024×1024高质量图像


### 6.3 StyleGAN v3（2021）


- 解决GAN的"纹理粘连"问题（纹理随图像旋转而不随物体移动）
- 引入非对齐生成（Alias-free generation）
- 傅里叶特征和连续信号处理


| 模型 | 核心创新 | 应用 |
| --- | --- | --- |
| ProGAN | 渐进式训练（4×4→1024×1024逐层增长） | 高分辨率生成 |
| SAGAN | 自注意力机制替代部分卷积 | 全局一致性生成 |
| BigGAN | 大规模GAN + 类别嵌入 + 截断技巧 | ImageNet级类别生成 |
| Progressive Distillation | 知识蒸馏加速GAN推理 | 高效推理 |
| GauGAN/SPADE | 语义布局→逼真图像 | 语义图像合成 |
| StarGAN v2 | 多域图像翻译 | 人脸属性编辑 |
| ESRGAN | 超分辨率GAN | 图像超分 |
| Talking Head | 面部动画生成 | 虚拟人驱动 |


<!-- Converted from: 01_GAN原理与变体.html -->
