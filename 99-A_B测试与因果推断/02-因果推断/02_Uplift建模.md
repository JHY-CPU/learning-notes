# Uplift建模


## 一、Uplift 建模的目标


传统机器学习预测的是"用户会不会转化"，而Uplift建模预测的是"这个干预会使用户的转化概率提升多少"。即估计**个体处理效应（ITE, Individual Treatment Effect）**。


$$
ITE(xi) = Yi(1) - Yi(0)
                传统模型：P(Y=1 | X=x) — 用户的转化概率
                Uplift模型：P(Y=1 | X=x, T=1) - P(Y=1 | X=x, T=0) — 处理带来的增量效应
$$


### 1.1 为什么需要 Uplift 建模


| 场景 | 问题 | Uplift的价值 |
| --- | --- | --- |
| 优惠券发放 | 不发优惠券也会买的用户，发了只是浪费 | 识别"只因优惠券才买"的用户 |
| 用户召回 | 自然会回来的用户不需要干预 | 找到"因干预才会回来"的用户 |
| 交叉销售 | 本来就会买的用户不需要推荐 | 找到"因推荐才购买"的用户 |


> **Important:** **核心洞察：**
> 用户的画像可以分为四类：
>
> - **Sure Things：**
>    不管有没有干预都会转化（干预浪费）
> - **Lost Causes：**
>    不管有没有干预都不会转化（干预浪费）
> - **Persuadables：**
>    只有干预才会转化（干预有效！）
> - **Sleeping Dogs：**
>    不干预会转化，干预反而不转化（干预有害！）
>
> Uplift建模的目标就是识别 Persuadables，避免 Sleeping Dogs。


## 二、Uplift 建模方法


### 2.1 T-Learner（Two-Model）


最简单的方法：分别在处理组和控制组上训练一个模型，然后取预测差异。


$$
τ̂(x) = ˆf1(x) - ˆf0(x)
                其中 ˆf1 用处理组数据训练，ˆf0 用控制组数据训练
$$


- **优点：**
   简单直观，可使用任何ML模型
- **缺点：**
   两个模型的误差可能放大uplift估计的方差


### 2.2 S-Learner（Single-Model）


将处理指示变量 T 作为特征之一，训练单个模型。


$$
τ̂(x) = ˆf(x, T=1) - ˆf(x, T=0)
$$


- **优点：**
   简单，共享数据训练
- **缺点：**
   模型可能忽略 T（尤其当其他特征很强时），uplift估计接近0


### 2.3 X-Learner


Kunzel et al. (2019) 提出的方法，利用交叉估计：


1. 在两组分别训练模型 ˆf
   ~1~
   和 ˆf
   ~0~
2. 计算伪处理效应：D
   ~i~
   ^1^
   = Y
   ~i~
   - ˆf
   ~0~
   (X
   ~i~
   )（处理组），D
   ~i~
   ^0^
   = ˆf
   ~1~
   (X
   ~i~
   ) - Y
   ~i~
   （控制组）
3. 分别在两组上训练uplift模型 ˆτ
   ~1~
   和 ˆτ
   ~0~
4. 加权组合：τ̂(x) = g(x)ˆτ
   ~1~
   (x) + (1-g(x))ˆτ
   ~0~
   (x)


### 2.4 R-Learner（Robust Learner）


Nie & Wager (2020) 提出，基于双重稳健（Doubly Robust）思想，使用残差回归：


$$
minτ ∑ [(Yi - ˆf-i(Xi)) - (Ti - ˆe-i(Xi))τ(Xi)]
$$


| 方法 | 模型数 | 复杂度 | 性能 |
| --- | --- | --- | --- |
| T-Learner | 2 | 低 | 中等 |
| S-Learner | 1 | 最低 | 通常较差 |
| X-Learner | 2+2 | 中等 | 较好，尤其样本不平衡 |
| R-Learner | 1（复合） | 较高 | 理论最优 |


## 三、Uplift 评估指标


### 3.1 Uplift Curve（提升曲线）


按预测的uplift从高到低排序用户，逐步纳入，计算累计的增量效应。


### 3.2 Qini Curve（基尼曲线）


Qini系数是Uplift建模最常用的评估指标。


$$
Qini(k) = ntconv(k) - nt(k) × ncconv(k) / nc(k)
                ntconv(k)：前k个用户中处理组的转化数
                nt(k)：前k个用户中处理组的人数
                Qini系数 = Qini曲线下的面积（相对于随机基线）
$$


> **Note:** **解读：**
> Qini曲线越高，模型越好——说明模型能更有效地识别出高uplift用户。随机策略的Qini曲线是一条直线。


## 四、Python 实战：CausalML Uplift 建模


> **Example:** ### 示例：电商优惠券Uplift建模
>
>
> ```
> import numpy as np
> import pandas as pd
> from sklearn.model_selection import train_test_split
> from causalml.inference.tree import UpliftRandomForestClassifier
> from causalml.inference.meta import BaseTClassifier, BaseXClassifier
> from causalml.metrics import plot_qini, qini_score
> import matplotlib.pyplot as plt
>
> # 1. 生成模拟数据
> np.random.seed(42)
> n = 5000
>
> # 用户特征
> age = np.random.normal(35, 10, n)
> income = np.random.normal(5000, 2000, n)
> purchase_history = np.random.poisson(5, n)
> is_member = np.random.binomial(1, 0.3, n)
>
> # 处理分配（随机化）
> treatment = np.random.binomial(1, 0.5, n)
>
> # 真实uplift与用户特征相关
> true_uplift = 0.1 + 0.05 * is_member - 0.01 * (age - 35)**2 / 100
> true_uplift = np.clip(true_uplift, -0.1, 0.3)
>
> # 转化概率
> base_prob = 0.2 + 0.01 * purchase_history - 0.0001 * income
> conversion_prob = base_prob + treatment * true_uplift
> conversion = np.random.binomial(1, np.clip(conversion_prob, 0, 1))
>
> df = pd.DataFrame({
>     'age': age, 'income': income,
>     'purchase_history': purchase_history,
>     'is_member': is_member,
>     'treatment': treatment,
>     'conversion': conversion
> })
>
> X = df[['age', 'income', 'purchase_history', 'is_member']].values
> y = df['conversion'].values
> w = df['treatment'].values
>
> # 划分数据
> X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
>     X, y, w, test_size=0.3, random_state=42)
>
> # 2. T-Learner
> t_learner = BaseTClassifier(learner=None)  # 默认GradientBoosting
> t_learner.fit(X_train, w_train, y_train)
> uplift_t = t_learner.predict(X_test)
>
> # 3. X-Learner
> x_learner = BaseXClassifier(learner=None)
> x_learner.fit(X_train, w_train, y_train)
> uplift_x = x_learner.predict(X_test)
>
> # 4. Uplift Random Forest
> uplift_rf = UpliftRandomForestClassifier(
>     n_estimators=100,
>     max_depth=5,
>     min_samples_leaf=100,
>     control_name=0
> )
> uplift_rf.fit(X_train, w_train, y_train)
> uplift_rf_pred = uplift_rf.predict(X_test)
>
> # 5. 评估
> df_test = pd.DataFrame(X_test, columns=['age', 'income', 'purchase_history', 'is_member'])
> df_test['treatment'] = w_test
> df_test['conversion'] = y_test
> df_test['uplift_t'] = uplift_t.flatten()
> df_test['uplift_x'] = uplift_x.flatten()
> df_test['uplift_rf'] = uplift_rf_pred.flatten()
>
> # 6. 绘制Qini曲线
> fig, ax = plt.subplots(figsize=(8, 6))
> for name, col in [('T-Learner', 'uplift_t'), ('X-Learner', 'uplift_x'),
>                    ('Uplift RF', 'uplift_rf')]:
>     plot_qini(df_test['conversion'].values,
>               df_test[col].values,
>               df_test['treatment'].values,
>               ax=ax, label=name)
>
> ax.set_title('Qini Curve Comparison')
> ax.legend()
> plt.tight_layout()
> plt.savefig("qini_curve.png", dpi=150)
> plt.show()
>
> # 7. Qini分数
> from causalml.metrics import auuc_score
> for name, col in [('T-Learner', 'uplift_t'), ('X-Learner', 'uplift_x'),
>                    ('Uplift RF', 'uplift_rf')]:
>     score = auuc_score(df_test[['treatment', 'conversion', col]].values)
>     print(f"{name} AUUC: {score:.4f}")
>
> # 8. 业务解读：找Top 20%高uplift用户
> top_20_pct = int(len(df_test) * 0.2)
> top_users = df_test.nlargest(top_20_pct, 'uplift_rf')
> print(f"\n=== Top 20% 高Uplift用户 ===")
> print(f"平均年龄: {top_users['age'].mean():.1f}")
> print(f"平均收入: {top_users['income'].mean():.0f}")
> print(f"会员比例: {top_users['is_member'].mean():.2f}")
> print(f"平均预测Uplift: {top_users['uplift_rf'].mean():.4f}")
> ```


## 五、电商营销应用


### 5.1 优惠券精准发放


> **Example:** **场景：**
> 某电商平台有100万用户，优惠券面额10元，想通过Uplift建模精准发放优惠券。
>
>
>
>
> **传统方式：**
> 给所有人发券 → 成本1000万元，其中大部分给了Sure Things（本来就会买的用户）。
>
>
>
>
> **Uplift方式：**
> 只给Top 20%高Uplift用户发券 → 成本200万元，精准触达Persuadables。
>
>
>
>
> **效果：**
> 假设Uplift模型识别的用户增量转化率为15%，20万人带来3万增量订单，每单利润50元 → 增量利润150万元，ROI = 150/200 = 75%（vs 随机发放ROI可能为负）。


### 5.2 实施流程


1. **数据准备：**
   历史实验数据（处理组+控制组）+ 用户特征
2. **模型训练：**
   用T-Learner/X-Learner/Uplift RF训练
3. **预测打分：**
   对全量用户预测uplift值
4. **分层运营：**
   按uplift值分层，对高uplift用户采取行动
5. **效果验证：**
   A/B测试验证uplift模型的效果


## 六、Uplift 建模的挑战


| 挑战 | 描述 | 应对策略 |
| --- | --- | --- |
| 反事实不可观测 | 每个用户只能看到一种结果 | 依赖随机实验数据 |
| 噪声大 | ITE的方差比预测任务大得多 | 增大样本量、使用元学习器 |
| 评估困难 | 无法直接计算真实ITE | 用Qini/AUUC间接评估 |
| 数据要求高 | 需要处理组和控制组的配对数据 | 利用历史A/B测试数据 |
| 效应异质性假设 | 假设uplift随特征变化 | 先做ATE分析确认异质性 |


## 总结


- Uplift建模预测的是处理带来的增量效应（ITE），而非绝对转化概率
- 用户可分为Sure Things、Lost Causes、Persuadables和Sleeping Dogs四类
- T-Learner最简单，X-Learner在样本不平衡时表现好，R-Learner理论最优
- Qini曲线和AUUC是最常用的评估指标
- 电商营销中，Uplift建模能显著提升ROI，精准识别值得干预的用户
- 实施Uplift建模的前提是拥有随机实验数据（A/B测试）


<!-- Converted from: 02_Uplift建模.html -->
