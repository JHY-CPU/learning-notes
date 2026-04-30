# 58_序列标注与 CRF 条件随机场

## 核心概念
- **序列标注 (Sequence Labeling)**：为输入序列的每个元素（token）分配一个标签。经典应用包括词性标注（POS Tagging）、命名实体识别（NER）、组块分析（Chunking）。
- **条件随机场 (Conditional Random Field, CRF)**：由 Lafferty et al. (2001) 提出，是一种判别式概率图模型。在序列标注中，CRF 对标签序列之间的依赖关系建模。
- **线性链 CRF (Linear-chain CRF)**：最常见的 CRF 形式，假设标签序列满足一阶马尔可夫性质——当前标签只依赖于前一标签和全局观测序列。
- **发射分数 (Emission Score)**：给定观测值 $x_i$，标签 $y_i$ 的得分。通常由神经网络（如 Bi-LSTM 或 BERT）输出。
- **转移分数 (Transition Score)**：从标签 $y_{i-1}$ 转移到 $y_i$ 的得分。CRF 模型学习哪些标签转移是合理的（如 I-PER 不能直接跟在 B-LOC 后面）。
- **维特比解码 (Viterbi Decoding)**：在推理时，在给定 CRF 参数下找到最可能的标签序列。利用动态规划在 $O(n \times |L|^2)$ 时间内找到全局最优序列。
- **与 Softmax 的关系**：没有 CRF 时，每个位置的标签独立预测（softmax 分类器），可能导致"B-LOC I-PER I-LOC"这种不合理的标签序列。CRF 通过转移分数约束这种不合理跳转。

## 数学推导
**线性链 CRF**：

给定观测序列 $\mathbf{x} = (x_1, \ldots, x_n)$ 和标签序列 $\mathbf{y} = (y_1, \ldots, y_n)$：

$$
P(\mathbf{y} | \mathbf{x}) = \frac{\exp\left(\sum_{i=1}^{n} E_{i, y_i} + \sum_{i=2}^{n} T_{y_{i-1}, y_i}\right)}{\sum_{\mathbf{y}'} \exp\left(\sum_{i=1}^{n} E_{i, y'_i} + \sum_{i=2}^{n} T_{y'_{i-1}, y'_i}\right)}
$$

其中 $E_{i, y_i}$ 是发射分数（来自 Bi-LSTM/BERT），$T_{y_{i-1}, y_i}$ 是转移分数。

**训练损失**（负对数似然）：
$$
\mathcal{L} = -\log P(\mathbf{y}^* | \mathbf{x}) = -\sum_{i=1}^n E_{i, y^*_i} - \sum_{i=2}^n T_{y^*_{i-1}, y^*_i} + \log Z(\mathbf{x})
$$

其中 $Z(\mathbf{x}) = \sum_{\mathbf{y}'} \exp(\sum_i E_{i, y'_i} + \sum_i T_{y'_{i-1}, y'_i})$ 是配分函数，可使用前向-后向算法高效计算。

**维特比解码**：
$$
y_i^* = \arg\max_{y_i} \left(\max_{\mathbf{y}_{<i}} \sum_{j=1}^{i-1} (E_{j, y_j} + T_{y_{j-1}, y_j}) + E_{i, y_i} + \max_{\mathbf{y}_{>i}} \sum_{j=i+1}^{n} (E_{j, y_j} + T_{y_{j-1}, y_j})\right)
$$

## 直观理解
- **CRF 像"标签语法检查"**：没有 CRF 时，模型在每个位置独立选择标签，可能产生"O B-PERSON I-LOCATION"这种不合理序列。CRF 相当于给标签序列加了一层"语法约束"——学会了"I-XXX 只能跟在 B-XXX 或 I-XXX 后面"等规则。
- **转移矩阵的含义**：在学习完成后，查看 CRF 的转移矩阵可以理解模型学到的"标签世界规则"。例如 $T_{[\text{B-PER} \to \text{I-PER}]}$ 很高，$T_{[\text{O} \to \text{I-PER}]}$ 很低（因为不能凭空出现 I-PER）。
- **全局最优 vs 局部最优**：Softmax 在每个位置独立做"局部最优"决策，CRF 寻找"全局最优"标签序列。就像拼图——一个一个放可能看起来都对，但最后发现整幅图对不上。CRF 像是先看整体再拼。
- **维特比解码**：虽然不是在所有可能的序列中穷举（指数级），但利用动态规划可以高效找到全局最优——就像在地图上有 N 个路口，维特比帮你找到了从起点到终点的最优路径。

## 代码示例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRFLayer(nn.Module):
    """CRF 层（简化实现）"""
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        # 转移分数 [from_label, to_label]
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels) * 0.01)
        # 不能从 START 转移到非起始标签等约束
        self.transitions.data[0, :] = -10000  # START -> others (index 0 = START)

    def forward_loss(self, emission_scores, tags, mask):
        """计算 CRF 负对数似然损失"""
        batch_size, seq_len, num_labels = emission_scores.shape
        # 真实标签序列的分数
        score = self._score_sentence(emission_scores, tags, mask)
        # 配分函数（所有可能序列的分数之和）
        Z = self._forward_algorithm(emission_scores, mask)
        # 损失 = Z - score（负对数似然）
        return (Z - score).mean()

    def _score_sentence(self, emissions, tags, mask):
        """计算一个标签序列的分数"""
        batch_size, seq_len = tags.shape
        score = torch.zeros(batch_size, device=emissions.device)
        for i in range(seq_len):
            # 加上当前标签的发射分数
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]
            if i > 0:
                # 加上转移分数
                score += self.transitions[tags[:, i-1], tags[:, i]] * mask[:, i]
        return score

    def _forward_algorithm(self, emissions, mask):
        """前向算法计算配分函数 log Z"""
        batch_size, seq_len, _ = emissions.shape
        # 初始化：从 START 出发
        alpha = emissions[:, 0] + self.transitions[0].unsqueeze(0)  # (batch, num_labels)
        alpha = torch.logsumexp(alpha, dim=1, keepdim=True)  # 简化版
        return alpha.sum()

    def decode(self, emission_scores, mask):
        """维特比解码"""
        batch_size, seq_len, num_labels = emission_scores.shape
        # 维特比解码
        scores = emissions[:, 0]
        backpointers = []
        for i in range(1, seq_len):
            next_scores = scores.unsqueeze(2) + self.transitions.unsqueeze(0)
            next_scores = next_scores + emissions[:, i].unsqueeze(1)
            best_scores, best_tags = next_scores.max(dim=1)
            scores = best_scores
            backpointers.append(best_tags)
        # 回溯
        best_path = [scores.argmax(dim=1)]
        for bp in reversed(backpointers):
            best_path.insert(0, bp.gather(1, best_path[0].unsqueeze(1)).squeeze(1))
        return torch.stack(best_path, dim=1)

# 使用 Bi-LSTM + CRF 进行 NER 的示例
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.bilstm = nn.LSTM(embed_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.emission = nn.Linear(hidden_size, num_labels)
        self.crf = CRFLayer(num_labels)

    def forward(self, x, labels=None, mask=None):
        emb = self.embed(x)
        lstm_out, _ = self.bilstm(emb)
        emissions = self.emission(lstm_out)  # (batch, seq, num_labels)
        if labels is not None:
            return self.crf.forward_loss(emissions, labels, mask)
        else:
            return self.crf.decode(emissions, mask)

# 演示
vocab_size, embed_size, hidden_size, num_labels = 10000, 128, 256, 9  # BIO 标签
model = BiLSTM_CRF(vocab_size, embed_size, hidden_size, num_labels)
x = torch.randint(0, 10000, (4, 20))  # (batch, seq_len)
labels = torch.randint(0, 9, (4, 20))
mask = torch.ones(4, 20, dtype=torch.bool)

loss = model(x, labels, mask)
print(f"CRF 损失: {loss.item():.4f}")

# 解码
predictions = model(x)
print(f"CRF 解码结果形状: {predictions.shape}")  # (4, 20)
```

## 深度学习关联
- **从 CRF 到 Transformer 的演进**：早期 SOTA 序列标注模型是 Bi-LSTM + CRF。BERT 出现后，发现只用 BERT 的 softmax 分类器（不 CRF）就已经超越了 Bi-LSTM + CRF 的效果。在预训练大语言模型时代，CRF 的使用减少，因为大规模预训练已经隐式学习了标签依赖关系。
- **CRF 在现代模型中的角色**：尽管在大多数序列标注任务中纯 BERT 已经足够，但在数据稀缺场景、小规模训练数据或需要严格标签约束的场景中，CRF 仍然有价值。一些研究表明 BERT + CRF 在某些场景下略优于纯 BERT。
- **结构化预测的泛化**：CRF 是"结构化预测"的经典方法之一。这一思想延展到其他结构化预测任务中——如句法分析、机器翻译的全局优化（Minimum Bayes Risk 解码）、以及蛋白质结构预测等跨领域任务。
