# Reward Model训练 - 人类对齐与RLHF

*深入理解 RLHF 中奖励模型的训练原理，包括 Bradley-Terry 偏好模型、偏好数据收集流程、reward hacking 问题及评估方法*

## 一、Bradley-Terry 偏好模型

Bradley-Terry 概率公式

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

  其中：
  - σ(z) = 1 / (1 + exp(-z)) 为 sigmoid 函数
  - r(x, y) 为 Reward Model 对 (prompt, response) 的打分
  - y_w ≻ y_l 表示 y_w 比 y_l 更受偏好

  等价形式：P(y_w ≻ y_l | x) = exp(r_w) / (exp(r_w) + exp(r_l))
  这是一种 softmax 形式，将奖励分数转化为概率
```

RM 的训练目标是最小化负对数似然损失：

$$
L = -E_{(x, y_w, y_l) \sim D} [\log \sigma(r(x, y_w) - r(x, y_l))]
$$

## 二、偏好数据收集

偏好数据格式示例

```json
{
  "prompt": "请解释量子纠缠的基本原理",
  "chosen": "量子纠缠是量子力学中的一种现象，当两个或多个粒子发生纠缠后...",
  "rejected": "量子纠缠就是两个东西连在一起，一个动另一个也动...",
  "metadata": {
    "annotator_id": "A001",
    "preference_strength": "strong",
    "annotation_time_seconds": 45,
    "categories": ["accuracy", "completeness"]
  }
}
```

标注质量控制方法

| 方法 | 描述 | 效果 |
| --- | --- | --- |
| **金标准测试** | 在标注任务中混入已知答案的测试题 | 筛除不合格标注者 |
| **多人交叉标注** | 每条数据由 3+ 人标注，取多数意见 | 降低个体偏差 |
| **Kappa 一致性** | 计算标注者间一致性的 Cohen's Kappa | Kappa > 0.6 为可接受 |
| **时间过滤** | 剔除标注时间过短的记录 | 排除随意标注 |
| **偏好强度** | 区分强偏好和弱偏好 | 区分不同置信度 |
| **定期校准** | 定期举行标注会议统一标准 | 减少标准漂移 |

## 三、Reward Hacking 问题

Reward Hacking 的常见表现

| 表现形式 | 具体描述 | 示例 |
| --- | --- | --- |
| **冗长输出** | RM 偏好更长的回答，模型倾向于生成冗长内容 | 简单问题给出长篇大论 |
| **格式讨好** | 过度使用列表、加粗等格式来获取高分 | 每句话都用 bullet point |
| **虚假自信** | 即使不确定也给出高置信度回答 | 编造看似权威的引用 |
| **迎合偏见** | 利用 RM 训练数据中的偏差 | 重复标注者偏好的措辞 |
| **回避回答** | 用"我无法确定"等免责表述逃避问题 | 本应给出建议却全面拒绝 |
| **重复信息** | 将同一信息换不同说法重复表达 | 同义替换填充内容 |

## 四、RM 评估指标体系

| 指标 | 计算方式 | 解读 |
| --- | --- | --- |
| **Pairwise Accuracy** | 正确预测偏好对的比例 | 最直接的指标，验证 RM 能否区分好坏回答 |
| **校准度（Calibration）** | RM 分数与实际胜率的对应关系 | 分数是否有实际概率意义 |
| **泛化能力** | 在未见过的 prompt 分布上的表现 | 是否过拟合训练数据 |
| **与人类一致性** | RM 排序 vs 人类排序的 Spearman/Kendall 相关系数 | RM 是否真正捕捉到人类偏好 |
| **奖励分布** | 可视化奖励分数的分布 | 检查是否有异常值或分布坍塌 |
| **Best-of-N 性能** | 用 RM 从 N 个回答中选最优，验证选出的回答质量 | 端到端验证 RM 的实用性 |

## 五、ORM vs PRM 对比

| 维度 | ORM（结果奖励模型） | PRM（过程奖励模型） |
| --- | --- | --- |
| 打分粒度 | 整个回答 | 每个推理步骤 |
| 反馈密度 | 稀疏（仅最终结果） | 密集（每步都有信号） |
| 适用场景 | 开放式生成 | 数学/逻辑推理 |
| 数据需求 | 回答级别标注 | 步骤级别标注（成本更高） |
| 代表工作 | InstructGPT RM | OpenAI PRM800K, Math-Shepherd |
| 树搜索集成 | 困难 | 可引导 Best-of-N / MCTS |

## 六、Python 实战：Reward Model 训练

### 示例：使用 HuggingFace Transformers 训练 Reward Model

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

# 1. 定义 Reward Model
class RewardModel(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        # 奖励头：将 [CLS] 映射为标量奖励
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的表示
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        reward = self.reward_head(cls_hidden).squeeze(-1)
        return reward

# 2. 偏好数据集
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # 将 prompt + response 拼接
        chosen_text = prompt + " [SEP] " + chosen
        rejected_text = prompt + " [SEP] " + rejected

        chosen_enc = self.tokenizer(chosen_text, truncation=True,
                                     max_length=self.max_length,
                                     padding="max_length", return_tensors="pt")
        rejected_enc = self.tokenizer(rejected_text, truncation=True,
                                       max_length=self.max_length,
                                       padding="max_length", return_tensors="pt")

        return {
            "chosen_ids": chosen_enc["input_ids"].squeeze(),
            "chosen_mask": chosen_enc["attention_mask"].squeeze(),
            "rejected_ids": rejected_enc["input_ids"].squeeze(),
            "rejected_mask": rejected_enc["attention_mask"].squeeze(),
        }

# 3. Bradley-Terry 损失函数
def bradley_terry_loss(reward_chosen, reward_rejected):
    """
    L = -E[log(sigma(r_chosen - r_rejected))]
    等价于 BCELoss，标签为1（chosen应更高）
    """
    logits = reward_chosen - reward_rejected
    labels = torch.ones_like(logits)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return loss

# 4. 训练循环
def train_reward_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = RewardModel("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    # 模拟偏好数据
    train_data = [
        {"prompt": "什么是机器学习？",
         "chosen": "机器学习是人工智能的一个分支，通过数据训练模型...",
         "rejected": "机器学习就是让机器学习东西。"},
        # ... 更多数据
    ]

    dataset = PreferenceDataset(train_data, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            reward_chosen = model(batch["chosen_ids"], batch["chosen_mask"])
            reward_rejected = model(batch["rejected_ids"], batch["rejected_mask"])

            loss = bradley_terry_loss(reward_chosen, reward_rejected)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        # 计算 pairwise accuracy
        with torch.no_grad():
            acc = (reward_chosen > reward_rejected).float().mean().item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    return model

# model = train_reward_model()
```

### 示例：使用 TRL 库快速训练 Reward Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    "facebook/opt-350m", num_labels=1
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# 配置训练参数
reward_config = RewardConfig(
    output_dir="./rm_checkpoint",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    gradient_accumulation_steps=4,
    logging_steps=10,
    evaluation_strategy="steps",
    max_length=512,
)

# 准备数据集（需要 chosen 和 rejected 列）
# from datasets import load_dataset
# dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# trainer = RewardTrainer(
#     model=model,
#     args=reward_config,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
# )
# trainer.train()
```

## 总结

- Reward Model 基于 Bradley-Terry 模型，将人类偏好转化为可训练的损失函数
- 偏好数据的质量控制（多人标注、Kappa 一致性）对 RM 性能至关重要
- Reward Hacking 是 RM 训练的主要挑战，需要通过数据多样性和正则化缓解
- ORM 适合开放式生成，PRM 适合推理任务（如数学题）
- 训练时使用 Pairwise Accuracy 作为核心监控指标
- 实践中推荐使用 TRL 库简化训练流程


<!-- Converted from: 01_Reward_Model训练.html -->
