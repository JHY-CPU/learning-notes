# 8_TPE 与 Hyperband

## 1. TPE (Tree-structured Parzen Estimator)

### 1.1 核心思想

**TPE (Bergstra et al., 2011)** 不直接建模 $P(y|x)$，而是建模 $P(x|y)$：

1. 将观测分为"好"和"差"两组
2. 分别建模 $l(x) = P(x|y < y^*)$ 和 $g(x) = P(x|y \geq y^*)$
3. 选择使 $l(x)/g(x)$ 最大的 $x$

### 1.2 与GP-BO的区别

| 方面 | GP-BO | TPE |
|------|-------|-----|
| 代理模型 | $P(y\|x)$ | $P(x\|y)$ |
| 不确定性 | GP提供 | 隐式 |
| 条件空间 | 不擅长 | 擅长 |
| 实现 | gpytorch | Optuna/Hyperopt |

### 1.3 数学原理

$$\alpha_{TPE}(x) = \frac{l(x)}{g(x)}$$

其中：
$$l(x) = P(x | y < y^*) \quad \text{(top }\gamma\text{ 分位的观测)}$$
$$g(x) = P(x | y \geq y^*) \quad \text{(其余观测)}$$

最优 $x^* = \arg\max_x \frac{l(x)}{g(x)}$：概率在好组中高、在差组中低的配置。

### 1.4 实现

```python
class TPEOptimizer:
    """TPE优化器"""
    def __init__(self, search_space, gamma=0.25):
        self.search_space = search_space
        self.gamma = gamma  # "好"组的比例
        self.observations = []
    
    def suggest(self):
        """建议下一个配置"""
        if len(self.observations) < 10:
            return self._random_sample()
        
        # 按性能排序
        sorted_obs = sorted(self.observations, key=lambda x: x[1], reverse=True)
        n_good = max(1, int(len(sorted_obs) * self.gamma))
        
        good_configs = [obs[0] for obs in sorted_obs[:n_good]]
        bad_configs = [obs[0] for obs in sorted_obs[n_good:]]
        
        # 拟合 l(x) 和 g(x)（用KDE）
        l_model = self._fit_kde(good_configs)
        g_model = self._fit_kde(bad_configs)
        
        # 采样候选，选择 l(x)/g(x) 最大的
        candidates = [self._random_sample() for _ in range(24)]
        scores = []
        for c in candidates:
            log_ratio = l_model.logpdf(c) - g_model.logpdf(c)
            scores.append(log_ratio)
        
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def _fit_kde(self, configs):
        """用核密度估计拟合分布"""
        from scipy.stats import gaussian_kde
        configs = np.array(configs).T  # (D, N)
        return gaussian_kde(configs)
    
    def observe(self, config, score):
        self.observations.append((config, score))
    
    def _random_sample(self):
        x = []
        for dim in self.search_space:
            if dim['type'] == 'log_uniform':
                x.append(np.exp(np.random.uniform(
                    np.log(dim['low']), np.log(dim['high'])
                )))
            elif dim['type'] == 'uniform':
                x.append(np.random.uniform(dim['low'], dim['high']))
            elif dim['type'] == 'choice':
                x.append(np.random.choice(dim['values']))
        return x
```

## 2. Hyperband

### 2.1 核心思想

**Hyperband (Li et al., 2017)** 通过**早停 (Early Stopping)** 大幅减少计算：

```
核心观察: 训练1个epoch后表现很差的模型，大概率最终也很差。
策略: 用少量资源（少epoch）快速筛选，只让有前途的配置继续训练。
```

### 2.2 Successive Halving

Hyperband 基于 **Successive Halving (SHA)**：

```
配置数: n_0, n_1, n_2, ..., n_K
资源:   r_0, r_1, r_2, ..., r_K

每轮:
  1. 给所有存活配置分配 r_k 资源
  2. 评估每个配置
  3. 保留 top 1/η 个配置
  4. 资源 ×η，配置 ÷η
```

```python
def successive_halving(configs, max_budget, eta=3):
    """Successive Halving"""
    n = len(configs)
    budget = max_budget // (eta ** int(np.log(n) / np.log(eta)))
    
    while len(configs) > 1:
        # 评估所有配置
        scores = []
        for config in configs:
            score = evaluate(config, budget=budget)
            scores.append(score)
        
        # 保留 top 1/eta
        keep = max(1, len(configs) // eta)
        top_indices = np.argsort(scores)[-keep:]
        configs = [configs[i] for i in top_indices]
        
        # 增加预算
        budget = min(budget * eta, max_budget)
    
    return configs[0]
```

### 2.3 Hyperband 的核心：探索-利用权衡

Successive Halving 需要预先决定：用多少初始配置？

- 配置多 → 每个配置资源少 → 可能错过好的配置
- 配置少 → 每个配置资源多 → 探索不充分

Hyperband 解决方案：**运行多轮 SHA，每轮配置数不同**。

```python
def hyperband(max_budget, eta=3):
    """Hyperband算法"""
    # 计算需要运行几轮 SHA
    s_max = int(np.log(max_budget) / np.log(eta))
    B = (s_max + 1) * max_budget
    
    best_config = None
    best_score = -float('inf')
    
    for s in range(s_max, -1, -1):
        # 每轮SHA的初始配置数
        n = int(np.ceil(B / max_budget / (s + 1) * eta ** s))
        r = max_budget * eta ** (-s)
        
        # 初始化n个随机配置
        configs = [random_config() for _ in range(n)]
        
        # 运行SHA
        for i in range(s + 1):
            n_i = int(n * eta ** (-i))
            r_i = int(r * eta ** i)
            
            # 评估
            scores = [evaluate(cfg, budget=r_i) for cfg in configs]
            
            # 保留top
            keep = max(1, int(n_i / eta))
            top_idx = np.argsort(scores)[-keep:]
            configs = [configs[j] for j in top_idx]
        
        # 检查最终结果
        final_score = evaluate(configs[0], budget=max_budget)
        if final_score > best_score:
            best_score = final_score
            best_config = configs[0]
    
    return best_config, best_score
```

### 2.4 Hyperband 配置

| $s$ | 初始配置 $n$ | 最小预算 $r$ | 迭代轮数 |
|-----|-------------|-------------|----------|
| 0 | 81 | 1 | 5 |
| 1 | 27 | 3 | 4 |
| 2 | 9 | 9 | 3 |
| 3 | 3 | 27 | 2 |
| 4 | 1 | 81 | 1 |

## 3. BOHB (Bayesian Optimization + Hyperband)

### 3.1 核心思想

**BOHB (Falkner et al., 2018)** 将贝叶斯优化（TPE）与Hyperband结合：

- Hyperband 的早停框架
- TPE 替代随机采样选择配置

```
Hyperband 决定: 评估哪些配置、分配多少资源
TPE 决定: 选择什么配置
```

### 3.2 实现思路

```python
def bohb(max_budget, eta=3, tpe_optimizer=None):
    """BOHB = TPE + Hyperband"""
    s_max = int(np.log(max_budget) / np.log(eta))
    B = (s_max + 1) * max_budget
    
    if tpe_optimizer is None:
        tpe_optimizer = TPEOptimizer(search_space)
    
    for s in range(s_max, -1, -1):
        n = int(np.ceil(B / max_budget / (s + 1) * eta ** s))
        r = max_budget * eta ** (-s)
        
        # 用TPE选择配置（而非随机）
        configs = [tpe_optimizer.suggest() for _ in range(n)]
        
        for i in range(s + 1):
            n_i = int(n * eta ** (-i))
            r_i = int(r * eta ** i)
            
            scores = [evaluate(cfg, budget=r_i) for cfg in configs]
            
            # TPE观察结果
            for cfg, score in zip(configs, scores):
                tpe_optimizer.observe(cfg, score)
            
            keep = max(1, int(n_i / eta))
            top_idx = np.argsort(scores)[-keep:]
            configs = [configs[j] for j in top_idx]
    
    return tpe_optimizer.best_config()
```

## 4. 方法对比

| 方法 | 类型 | 早停 | 代理模型 | 条件空间 | 速度 |
|------|------|------|----------|----------|------|
| TPE | 贝叶斯优化 | 否 | KDE | 好 | 中 |
| Hyperband | 早停 | 是 | 无 | 一般 | 快 |
| BOHB | 混合 | 是 | KDE | 好 | 快 |
| Random Search | 随机 | 否 | 无 | 好 | 快 |
| GP-BO | 贝叶斯优化 | 否 | GP | 差 | 慢 |

## 5. 实践建议

| 场景 | 推荐方法 |
|------|----------|
| 单次评估便宜 | TPE |
| 单次评估昂贵 | BOHB / Hyperband |
| 条件超参数多 | TPE / BOHB |
| 并行评估 | BOHB |
| 快速基线 | Hyperband |

---

**关键要点**：
1. TPE 建模 $P(x|y)$ 而非 $P(y|x)$，用 $l(x)/g(x)$ 比率选择配置
2. Hyperband 通过早停大幅减少计算：差配置早期被淘汰
3. BOHB 结合了TPE的代理模型和Hyperband的早停框架，是效率最优的方法之一
4. TPE 天然支持条件超参数（如优化器特有参数）
