# 推荐系统原型 (Recommendation System)

## 项目需求与功能分析

推荐系统是互联网产品（电商、视频、音乐）的核心引擎。本项目实现基于协同过滤和矩阵分解的推荐算法原型，帮助理解推荐系统的底层原理。

### 核心功能

- 用户-物品评分矩阵管理
- 基于用户的协同过滤 (User-based CF)
- 基于物品的协同过滤 (Item-based CF)
- 矩阵分解推荐 (SVD / ALS)
- 相似度计算（余弦相似度、皮尔逊相关系数）
- 推荐结果评估（MAE、RMSE）

### 数据结构

用户-物品评分矩阵 R[m][n]，其中 R[i][j] 表示用户 i 对物品 j 的评分，未评分位置用 0 或 None 表示。

## 核心算法原理

### 协同过滤

**User-based CF**：找到与目标用户口味相似的用户，用他们的评分来预测。

**Item-based CF**：找到与目标物品相似的物品，用用户对相似物品的评分来预测。

### 矩阵分解

将稀疏的 m x n 评分矩阵分解为两个低秩矩阵：

```
R ≈ P x Q^T
```

其中 P 是 m x k 的用户隐因子矩阵，Q 是 n x k 的物品隐因子矩阵，k 是隐因子维度。

通过随机梯度下降 (SGD) 或交替最小二乘 (ALS) 优化。

## 完整代码实现

```python
import math
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class RatingMatrix:
    """用户-物品评分矩阵"""

    def __init__(self):
        self.ratings: Dict[Tuple[int, int], float] = {}
        self.users: set = set()
        self.items: set = set()

    def add_rating(self, user: int, item: int, rating: float):
        self.ratings[(user, item)] = rating
        self.users.add(user)
        self.items.add(item)

    def get_rating(self, user: int, item: int) -> Optional[float]:
        return self.ratings.get((user, item))

    def get_user_ratings(self, user: int) -> Dict[int, float]:
        return {item: r for (u, item), r in self.ratings.items() if u == user}

    def get_item_ratings(self, item: int) -> Dict[int, float]:
        return {user: r for (user, it), r in self.ratings.items() if it == item}

    def user_mean(self, user: int) -> float:
        urs = self.get_user_ratings(user)
        return sum(urs.values()) / len(urs) if urs else 0.0

    def to_matrix(self) -> List[List[float]]:
        um = max(self.users) + 1
        im = max(self.items) + 1
        matrix = [[0.0] * im for _ in range(um)]
        for (u, i), r in self.ratings.items():
            matrix[u][i] = r
        return matrix


class SimilarityCalculator:
    """相似度计算"""

    @staticmethod
    def cosine_similarity(vec_a: Dict[int, float], vec_b: Dict[int, float]) -> float:
        """余弦相似度"""
        common = set(vec_a.keys()) & set(vec_b.keys())
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def pearson_similarity(vec_a: Dict[int, float], vec_b: Dict[int, float]) -> float:
        """皮尔逊相关系数"""
        common = set(vec_a.keys()) & set(vec_b.keys())
        if len(common) < 2:
            return 0.0
        mean_a = sum(vec_a[k] for k in common) / len(common)
        mean_b = sum(vec_b[k] for k in common) / len(common)
        num = sum((vec_a[k] - mean_a) * (vec_b[k] - mean_b) for k in common)
        da = math.sqrt(sum((vec_a[k] - mean_a)**2 for k in common))
        db = math.sqrt(sum((vec_b[k] - mean_b)**2 for k in common))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)


class CollaborativeFilter:
    """协同过滤推荐"""

    def __init__(self, matrix: RatingMatrix, method='user'):
        self.matrix = matrix
        self.method = method
        self.sim_calc = SimilarityCalculator()

    def _user_similarity(self, u1: int, u2: int) -> float:
        return self.sim_calc.pearson_similarity(
            self.matrix.get_user_ratings(u1),
            self.matrix.get_user_ratings(u2)
        )

    def _item_similarity(self, i1: int, i2: int) -> float:
        return self.sim_calc.pearson_similarity(
            self.matrix.get_item_ratings(i1),
            self.matrix.get_item_ratings(i2)
        )

    def predict_user_based(self, user: int, item: int, k: int = 5) -> float:
        """User-based 预测评分"""
        # 找到对 item 有评分的用户
        item_users = self.matrix.get_item_ratings(item)
        if not item_users:
            return self.matrix.user_mean(user)

        # 计算相似度
        sims = []
        for other in item_users:
            if other == user:
                continue
            sim = self._user_similarity(user, other)
            sims.append((sim, other))

        sims.sort(reverse=True)
        top_k = sims[:k]

        if not top_k:
            return self.matrix.user_mean(user)

        # 加权平均
        num = sum(sim * (self.matrix.get_rating(other, item) - self.matrix.user_mean(other))
                   for sim, other in top_k)
        den = sum(abs(sim) for sim, _ in top_k)

        if den == 0:
            return self.matrix.user_mean(user)

        return self.matrix.user_mean(user) + num / den

    def predict_item_based(self, user: int, item: int, k: int = 5) -> float:
        """Item-based 预测评分"""
        user_items = self.matrix.get_user_ratings(user)
        if not user_items:
            return 3.0

        sims = []
        for other_item in user_items:
            if other_item == item:
                continue
            sim = self._item_similarity(item, other_item)
            sims.append((sim, other_item))

        sims.sort(reverse=True)
        top_k = sims[:k]

        if not top_k:
            return 3.0

        num = sum(sim * user_items[item_j] for sim, item_j in top_k)
        den = sum(abs(sim) for sim, _ in top_k)

        if den == 0:
            return 3.0

        return num / den

    def recommend(self, user: int, n: int = 5, k: int = 5) -> List[Tuple[int, float]]:
        """为用户推荐 top-n 物品"""
        rated = set(self.matrix.get_user_ratings(user).keys())
        all_items = self.matrix.items
        candidates = all_items - rated

        predictions = []
        for item in candidates:
            if self.method == 'user':
                pred = self.predict_user_based(user, item, k)
            else:
                pred = self.predict_item_based(user, item, k)
            predictions.append((item, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class MatrixFactorization:
    """矩阵分解推荐 (SGD)"""

    def __init__(self, matrix: RatingMatrix, k: int = 10,
                 lr: float = 0.01, reg: float = 0.02):
        self.matrix = matrix
        self.k = k
        self.lr = lr
        self.reg = reg
        self.um = max(matrix.users) + 1
        self.im = max(matrix.items) + 1
        # 随机初始化
        self.P = [[random.gauss(0, 0.1) for _ in range(k)] for _ in range(self.um)]
        self.Q = [[random.gauss(0, 0.1) for _ in range(k)] for _ in range(self.im)]
        self.user_bias = [0.0] * self.um
        self.item_bias = [0.0] * self.im
        self.global_mean = sum(matrix.ratings.values()) / len(matrix.ratings)

    def train(self, epochs: int = 100):
        """SGD 训练"""
        for epoch in range(epochs):
            total_error = 0
            items = list(self.matrix.ratings.items())
            random.shuffle(items)

            for (u, i), r in items:
                pred = self.predict(u, i)
                err = r - pred
                total_error += err ** 2

                # 更新偏置
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

                # 更新隐因子
                for f in range(self.k):
                    puf = self.P[u][f]
                    qif = self.Q[i][f]
                    self.P[u][f] += self.lr * (err * qif - self.reg * puf)
                    self.Q[i][f] += self.lr * (err * puf - self.reg * qif)

            rmse = math.sqrt(total_error / len(self.matrix.ratings))
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: RMSE = {rmse:.4f}")

    def predict(self, user: int, item: int) -> float:
        pred = self.global_mean + self.user_bias[user] + self.item_bias[item]
        pred += sum(self.P[user][f] * self.Q[item][f] for f in self.k)
        return max(1.0, min(5.0, pred))  # 限制在 1-5 范围

    def recommend(self, user: int, n: int = 5) -> List[Tuple[int, float]]:
        rated = set(self.matrix.get_user_ratings(user).keys())
        candidates = self.matrix.items - rated
        preds = [(item, self.predict(user, item)) for item in candidates]
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n]


def evaluate(matrix: RatingMatrix, method, test_ratio=0.2):
    """评估推荐算法"""
    all_ratings = list(matrix.ratings.items())
    random.seed(42)
    random.shuffle(all_ratings)
    split = int(len(all_ratings) * (1 - test_ratio))
    train_data = all_ratings[:split]
    test_data = all_ratings[split:]

    # 重建训练矩阵
    train_matrix = RatingMatrix()
    for (u, i), r in train_data:
        train_matrix.add_rating(u, i, r)

    if method == 'user':
        cf = CollaborativeFilter(train_matrix, 'user')
        pred_fn = cf.predict_user_based
    elif method == 'item':
        cf = CollaborativeFilter(train_matrix, 'item')
        pred_fn = cf.predict_item_based
    else:
        mf = MatrixFactorization(train_matrix, k=5)
        mf.train(epochs=50)
        pred_fn = mf.predict

    errors = []
    for (u, i), actual in test_data:
        predicted = pred_fn(u, i)
        errors.append((actual - predicted) ** 2)

    mae = sum(abs(actual - pred_fn(u, i)) for (u,i), actual in test_data) / len(test_data)
    rmse = math.sqrt(sum(errors) / len(errors))
    return mae, rmse
```

## 测试用例

```python
import unittest

class TestRecommendation(unittest.TestCase):
    def setUp(self):
        self.rm = RatingMatrix()
        data = [
            (0,0,5),(0,1,3),(0,3,4),
            (1,0,4),(1,2,4),(1,3,3),
            (2,0,3),(2,1,2),(2,3,5),
            (3,0,2),(3,2,3),(3,3,4),
            (4,1,4),(4,2,5),(4,3,3),
        ]
        for u,i,r in data: self.rm.add_rating(u, i, r)

    def test_cf_recommend(self):
        cf = CollaborativeFilter(self.rm, 'user')
        recs = cf.recommend(0, n=2)
        self.assertEqual(len(recs), 2)
        for item, score in recs:
            self.assertIsInstance(score, float)

    def test_mf_train(self):
        mf = MatrixFactorization(self.rm, k=3)
        mf.train(epochs=10)
        pred = mf.predict(0, 2)
        self.assertGreaterEqual(pred, 1.0)
        self.assertLessEqual(pred, 5.0)

    def test_similarity_bounds(self):
        sc = SimilarityCalculator()
        a = {0: 5, 1: 3, 2: 4}
        b = {0: 4, 1: 2, 2: 5}
        cos = sc.cosine_similarity(a, b)
        self.assertGreater(cos, 0)
        self.assertLessEqual(cos, 1.0)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **混合推荐**：结合协同过滤和内容特征
2. **深度学习**：使用神经网络学习用户和物品嵌入
3. **冷启动处理**：新用户 / 新物品的推荐策略
4. **实时推荐**：在线学习更新模型
5. **隐式反馈**：处理点击、浏览等隐式行为
6. **多样性优化**：平衡准确性和推荐多样性
7. **A/B 测试框架**：对比不同推荐算法的效果
