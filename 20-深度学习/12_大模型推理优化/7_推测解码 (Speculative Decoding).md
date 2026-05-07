# 7_推测解码 (Speculative Decoding)

## 1. 核心思想

推测解码 (Leviathan et al., 2023) 的核心思想是：用**小模型快速生成草稿**，再用**大模型并行验证**，从而加速推理而不损失质量。

```
标准自回归生成 (大模型):
  Step 1: LLM(tokens₀) → token₁     (1次大模型调用)
  Step 2: LLM(tokens₀₁) → token₂    (1次大模型调用)
  Step 3: LLM(tokens₀₁₂) → token₃   (1次大模型调用)
  ...
  生成 N 个 token 需要 N 次大模型调用

推测解码:
  Step 1: 小模型快速生成 K 个草稿 token [d₁, d₂, ..., dₖ]
  Step 2: 大模型并行验证这 K 个 token
  Step 3: 接受匹配的 token，拒绝不匹配的
  生成 N 个 token 平均需要 N/K 次大模型调用
```

## 2. 数学原理

```
关键定理: 推测解码生成的分布与大模型完全一致

设:
  - p(x): 大模型的概率分布 (Target)
  - q(x): 小模型的概率分布 (Draft)

接受概率:
  α(x) = min(1, p(x) / q(x))

如果小模型生成 token x:
  - 以概率 α(x) 接受 x
  - 以概率 1-α(x) 拒绝 x，并从修正分布重新采样:
    p'(x) = max(0, p(x) - q(x)) / Z

这保证了最终的输出分布与大模型完全相同！
```

```python
import torch
import torch.nn.functional as F

def speculative_decode(target_logits, draft_tokens, draft_probs):
    """
    推测解码验证步骤

    target_logits: 大模型对草稿位置的 logits [K, vocab_size]
    draft_tokens: 小模型生成的草稿 token [K]
    draft_probs: 小模型在草稿位置的概率 [K, vocab_size]
    """
    K = len(draft_tokens)
    target_probs = F.softmax(target_logits, dim=-1)

    accepted = []
    for i in range(K):
        # 接受概率 = min(1, p_target / p_draft)
        p_target = target_probs[i, draft_tokens[i]]
        p_draft = draft_probs[i, draft_tokens[i]]
        accept_ratio = min(1.0, (p_target / p_draft).item())

        if torch.rand(1).item() < accept_ratio:
            # 接受草稿 token
            accepted.append(draft_tokens[i])
        else:
            # 拒绝，从修正分布采样
            adjusted = target_probs[i] - draft_probs[i]
            adjusted = adjusted.clamp(min=0)
            adjusted = adjusted / adjusted.sum()
            new_token = torch.multinomial(adjusted, 1).item()
            accepted.append(new_token)
            break  # 拒绝后停止

    # 额外采样一个 token（确保至少生成 K+1 个 token）
    if len(accepted) <= K:
        last_pos = len(accepted)
        extra_probs = target_probs[last_pos]
        extra_token = torch.multinomial(extra_probs, 1).item()
        accepted.append(extra_token)

    return accepted
```

## 3. 完整实现

```python
class SpeculativeDecoder:
    """推测解码器"""

    def __init__(self, target_model, draft_model, tokenizer,
                 num_speculative_tokens: int = 5):
        self.target = target_model  # 大模型
        self.draft = draft_model    # 小模型
        self.tokenizer = tokenizer
        self.K = num_speculative_tokens

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        tokens = self.tokenizer.encode(prompt)
        generated = 0
        total_draft = 0
        total_accepted = 0

        while generated < max_tokens:
            # 1. 草稿模型生成 K 个 token
            draft_tokens, draft_probs = self.draft_generate(tokens, self.K)
            total_draft += self.K

            # 2. 目标模型并行验证
            target_logits = self.target_forward(tokens + draft_tokens)

            # 3. 接受/拒绝
            accepted = speculative_decode(
                target_logits[-(self.K+1):],  # K+1 个位置的 logits
                draft_tokens,
                draft_probs
            )

            tokens.extend(accepted)
            generated += len(accepted)
            total_accepted += len(accepted) - 1  # -1 因为最后一个额外采样

            # 检查结束条件
            if tokens[-1] == self.tokenizer.eos_token_id:
                break

        acceptance_rate = total_accepted / max(total_draft, 1)
        print(f"接受率: {acceptance_rate:.1%}")
        print(f"加速比: 约 {1/(1-acceptance_rate+0.01):.1f}x")

        return self.tokenizer.decode(tokens)

    def draft_generate(self, prefix_tokens, K):
        """草稿模型快速生成 K 个 token"""
        draft_tokens = []
        draft_probs = []

        current_tokens = list(prefix_tokens)
        with torch.no_grad():
            for _ in range(K):
                logits = self.draft(current_tokens)
                probs = F.softmax(logits[-1], dim=-1)
                token = torch.multinomial(probs, 1).item()

                draft_tokens.append(token)
                draft_probs.append(probs)
                current_tokens.append(token)

        return draft_tokens, torch.stack(draft_probs)

    def target_forward(self, all_tokens):
        """目标模型一次前向传播，获取所有位置的 logits"""
        with torch.no_grad():
            logits = self.target(all_tokens)
        return logits
```

## 4. 加速比分析

```
加速比 = 实际接受的 token 数 / 目标模型调用次数

假设接受率为 r (每个草稿 token 被接受的概率):

期望接受 token 数 = K × r + 1 (最后额外采样一个)
目标模型调用次数 = 1 (并行验证 K 个 token 只需 1 次)

单步加速比 ≈ K × r + 1

实际加速比参考:
┌──────────┬───────────┬──────────────┐
│ 草稿大小 K │ 接受率 r  │  加速比       │
├──────────┼───────────┼──────────────┤
│    3     │   0.8     │   ~3.4x      │
│    5     │   0.7     │   ~4.5x      │
│    5     │   0.8     │   ~5.0x      │
│    5     │   0.9     │   ~5.5x      │
│   10     │   0.7     │   ~8.0x      │
│   10     │   0.9     │   ~10.0x     │
└──────────┴───────────┴──────────────┘

接受率取决于:
- 草稿模型与目标模型的相似度
- 任务的确定性 (代码 > 文学创作)
- 温度参数 (温度越低，接受率越高)
```

## 5. 草稿模型选择策略

```python
class DraftModelStrategies:
    """草稿模型选择策略"""

    # 策略 1: 独立小模型
    # 使用同系列的小版本 (如 LLaMA-7B 作为 LLaMA-70B 的草稿)
    independent_draft = {
        "target": "LLaMA-70B",
        "draft": "LLaMA-7B",
        "pros": "简单，高质量",
        "cons": "需要额外加载草稿模型"
    }

    # 策略 2: 自推测解码 (Self-Speculative Decoding)
    # 用目标模型的早期层作为草稿
    self_draft = {
        "target": "LLaMA-70B (所有层)",
        "draft": "LLaMA-70B (前 20 层)",
        "pros": "无需额外模型",
        "cons": "需要修改模型架构"
    }

    # 策略 3: Medusa (多头预测)
    # 在目标模型上添加多个解码头
    medusa = {
        "target": "LLaMA-7B + Medusa heads",
        "draft": "Medusa heads (并行预测 K 个 token)",
        "pros": "无需额外模型，并行预测",
        "cons": "需要微调 Medusa heads"
    }

    # 策略 4: EAGLE (基于特征的草稿)
    eagle = {
        "target": "LLaMA-70B",
        "draft": "EAGLE (使用目标模型特征)",
        "pros": "高接受率",
        "cons": "需要训练草稿网络"
    }
```

## 6. 实际使用 (HuggingFace)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载目标模型和草稿模型
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)
draft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

# 使用推测解码生成
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
inputs = tokenizer("量子计算的原理是", return_tensors="pt").to("cuda")

outputs = target_model.generate(
    **inputs,
    max_new_tokens=256,
    assistant_model=draft_model,  # 启用推测解码
    do_sample=True,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0]))
```

## 7. Medusa: 多头推测

```python
class MedusaHead(nn.Module):
    """Medusa 多预测头"""

    def __init__(self, hidden_size, vocab_size, num_heads=5):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size) for _ in range(num_heads)
        ])

    def forward(self, hidden_states):
        """并行预测未来 K 个 token"""
        predictions = []
        for head in self.heads:
            logits = head(hidden_states)
            predictions.append(logits)
        return predictions  # [K, batch, seq, vocab]

    def speculative_generate(self, hidden_states):
        """Medusa 推测生成"""
        predictions = self.forward(hidden_states)
        # 每个 head 预测一个位置
        draft_tokens = []
        for logits in predictions:
            probs = F.softmax(logits[-1], dim=-1)
            token = torch.multinomial(probs, 1)
            draft_tokens.append(token)
        return draft_tokens
```

## 总结

推测解码是一种**无损加速技术** -- 在不改变输出分布的前提下，通过"草稿+验证"机制实现 2-5x 加速。关键因素是**草稿质量**（接受率）和**草稿成本**。Medusa 和 EAGLE 等新方法通过改进草稿质量进一步提升了加速比。
