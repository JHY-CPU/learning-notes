# 60_Wav2Vec 2.0：自监督语音预训练

## 核心概念
- **Wav2Vec 2.0**：由 Facebook AI (2020) 提出，是一个自监督语音预训练框架。在无标注语音数据上预训练，然后在下游语音任务（语音识别、说话人识别等）上微调，大幅降低了对标注数据的需求。
- **自监督学习框架**：与 BERT 的 MLM 类似，Wav2Vec 2.0 在"隐藏"一些语音片段的条件下，预测被隐藏的量化表示。这是一个"对比学习"和"掩码预测"的结合。
- **特征编码器 (Feature Encoder)**：由多层 CNN 组成，将原始语音波形转换为隐式表示（频率约 50Hz，即每 20ms 一个向量）。
- **上下文编码器 (Context Network)**：使用 Transformer，对特征编码器的输出进行上下文建模。与 BERT 类似，它通过自注意力捕捉全局语音上下文。
- **量化模块 (Quantization Module)**：将特征编码器的输出离散化为有限数量的"码本"(codebook) 向量。这是学习目标中的"标签"生成器。
- **对比损失 (Contrastive Loss)**：鼓励模型从量化后的目标表示中选出被遮盖位置对应的正确表示，同时排斥其他负样本。
- **Gumbel-Softmax 量化**：使用 Gumbel-Softmax 实现离散化的可微分近似，使整个模型可以端到端训练。
- **训练策略**：先在大量无标注语音（如 LibriSpeech 960h 未转写音频）上预训练，再在有限标注数据上微调。仅需 10 分钟标注数据即可达到传统方法 100 小时训练的效果。

## 数学推导
**掩码策略**：随机遮盖特征编码器输出中约 50% 的时间步（每个掩码跨度约 10 个时间步）。

**上下文编码器输出**：
$$
c_t = \text{Transformer}(z_1, \ldots, z_T)_t
$$

其中 $z_t$ 是被遮盖（可能被替换为 $z_{\text{mask}}$）的特征编码器输出。

**对比学习目标**（每个被遮盖位置 $t$）：
$$
\mathcal{L}_t = -\log \frac{\exp(\text{sim}(c_t, q_t) / \kappa)}{\sum_{\tilde{q} \in Q_t} \exp(\text{sim}(c_t, \tilde{q}) / \kappa)}
$$

其中 $q_t$ 是位置 $t$ 的真实量化表示（正样本），$Q_t$ 包含正样本和 $K$ 个随机采样的负样本，$\text{sim}$ 是余弦相似度，$\kappa$ 是温度参数。

**多样性损失**：鼓励量化模块均衡使用码本中的每个向量：
$$
\mathcal{L}_{\text{diversity}} = -\frac{1}{G \cdot V} \sum_{g=1}^{G} \sum_{v=1}^{V} \bar{p}_{g,v} \log \bar{p}_{g,v}
$$

其中 $G$ 是码本组数，$V$ 是每组的向量数，$\bar{p}_{g,v}$ 是第 $g$ 组选择第 $v$ 个向量的平均概率。

**总损失**：
$$
\mathcal{L} = \mathcal{L}_{\text{contrastive}} + \alpha \mathcal{L}_{\text{diversity}}
$$

## 直观理解
- **Wav2Vec 2.0 像"语音版的 BERT"**：BERT 通过遮盖文本中的词来训练双向理解，Wav2Vec 2.0 通过遮盖语音片段来训练语音理解。两者都是"遮住一些东西，让模型猜"的自监督学习。
- **量化模块像"给语音贴标签"**：语音是连续的模拟信号，量化模块把它变成离散的"语音音素类别"。就像把颜色空间中的连续 RGB 值离散化为"红橙黄绿青蓝紫"——虽然简化了但保留了本质信息。
- **对比学习的直觉**：模型的任务是在一堆"干扰项"（负样本）中挑出正确的那个。就像考试中的四选一——即使不知道所有知识，能区分正确答案和错误答案也是有用的能力。
- **为什么自监督语音预训练重要**：语音数据的标注（语音转写为文字）成本极高——1 小时语音可能需要 4 小时人工标注。Wav2Vec 2.0 使得可以使用"无标注的语音"进行预训练，标注量从 1000 小时降低到 10 小时。

## 代码示例
```python
# Wav2Vec 2.0 使用示例
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# 加载 Wav2Vec 2.0 模型和处理器
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 模拟语音输入（实际应加载真实音频）
def simulate_speech_input(seq_length=16000):
    """生成模拟语音信号（16kHz, 1秒）"""
    return torch.randn(seq_length)

# 语音识别推理流程
def transcribe_speech(waveform, sample_rate=16000):
    # 1. 预处理：resample 到 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # 2. 特征提取
    inputs = processor(waveform, sampling_rate=16000, return_tensors='pt')

    # 3. 模型推理
    with torch.no_grad():
        logits = model(**inputs).logits

    # 4. 解码（CTC 贪婪解码）
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

# 模拟调用
print("Wav2Vec 2.0 语音识别演示")
print("模型: facebook/wav2vec2-base-960h")

# Wav2Vec 2.0 架构信息
print("\nWav2Vec 2.0 架构:")
print("  - 特征编码器: 7 层 CNN (将原始波形编码为 ~50Hz 表示)")
print("  - 量化模块: Gumbel-Softmax 码本 (320 个向量 × 2 组)")
print("  - 上下文编码器: 12 层 Transformer")
print("  - 预训练数据: LibriSpeech 960h 无标注语音")
print("  - 微调数据: LibriSpeech 100h 标注数据")
print("  - Word Error Rate: ~3.0% (LibriSpeech test-clean)")

# 演示掩码预测的概念
def mask_prediction_demo():
    """模拟 Wav2Vec 2.0 的掩码预测"""
    T = 100  # 时间步数
    d_model = 768
    
    # 模拟编码器输出
    z = torch.randn(T, d_model)
    
    # 遮盖 50% 的时间步
    mask = torch.rand(T) > 0.5
    z_masked = z.clone()
    z_masked[mask] = 0  # 遮盖
    
    print(f"\n掩码预测演示:")
    print(f"  总时间步: {T}")
    print(f"  被遮盖步数: {mask.sum().item()}")
    print(f"  遮盖比例: {mask.float().mean().item() * 100:.0f}%")
    print(f"  Transformer 任务: 根据未遮盖部分预测遮盖部分的量化表示")

mask_prediction_demo()
```

## 深度学习关联
- **自监督学习在语音领域的成功**：Wav2Vec 2.0 是自监督学习在语音领域的突破性工作，与 BERT (文本)、MAE (图像) 一起构成了三大模态的自监督预训练里程碑。
- **从 Wav2Vec 到 HuBERT 和 Whisper**：Wav2Vec 2.0 启发了后续的 HuBERT（使用聚类标签而非量化表示）和 Whisper（OpenAI 的有监督大规模语音模型），推动了语音处理的快速发展。
- **多模态语音理解的趋势**：Wav2Vec 2.0 的结合对比学习 + 掩码预测的思想正在向多模态扩展——如 AV-HuBERT（音视频联合）和 SpeechT5（语音-文本联合预训练）。
