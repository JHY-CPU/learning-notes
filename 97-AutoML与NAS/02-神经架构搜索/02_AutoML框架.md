# AutoML框架


## 一、AutoML 概述


AutoML（Automated Machine Learning）旨在自动化机器学习的全流程，包括特征工程、模型选择、超参数优化、模型评估等环节，降低ML的使用门槛。


### 1.1 AutoML 覆盖的环节


| 环节 | 自动化内容 | 人工参与 |
| --- | --- | --- |
| 数据预处理 | 缺失值填充、编码、标准化 | 数据收集和清洗 |
| 特征工程 | 特征选择、特征变换、交叉特征 | 领域特征定义 |
| 模型选择 | 从候选模型中选择最优 | 约束候选范围 |
| 超参数优化 | 搜索最优超参数 | 设定搜索范围 |
| 模型评估 | 交叉验证、性能报告 | 解读结果 |
| 模型部署 | 模型打包、API生成 | 集成到业务系统 |


## 二、AutoSklearn


AutoSklearn（Feurer et al., 2015）是基于 scikit-learn 的 AutoML 系统，使用贝叶斯优化和元学习来自动选择和配置机器学习管道。


### 2.1 核心特性


- **元学习（Meta-learning）：**
   利用历史数据集的经验来热启动搜索
- **集成选择（Ensemble Selection）：**
   自动组合多个模型形成集成
- **自动特征预处理：**
   One-hot编码、标准化、特征选择


### 2.2 使用示例


```
import autosklearn.classification
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=300,     # 总时间限制（秒）
    per_run_time_limit=30,           # 单模型时间限制
    n_jobs=-1,                       # 并行数
    memory_limit=4096,               # 内存限制（MB）
    ensemble_size=50,                # 集成模型数量
    initial_configurations_via_metalearning=25  # 元学习初始配置数
)

automl.fit(X_train, y_train)
print(f"测试准确率: {automl.score(X_test, y_test):.4f}")
print(automl.show_models())  # 查看最终集成的模型
```


> **Note:** **AutoSklearn 2.0 改进：**
> 引入了自适应采样和自适应配置，更加智能地分配计算预算，性能显著提升。


## 三、AutoGluon


AutoGluon 是 Amazon AWS 开发的 AutoML 框架，专注于**开箱即用的高精度预测**。


### 3.1 核心特性


- **多层堆叠集成：**
   自动训练多层模型并做stacking
- **自动特征工程：**
   自动处理文本、图像、表格数据
- **零配置高性能：**
   默认设置即可获得强大效果
- **快速原型：**
   几分钟内完成端到端ML


```
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# 加载数据
train_data = TabularDataset('train.csv')
test_data = TabularDataset('test.csv')

# 一行代码训练
predictor = TabularPredictor(label='target_column',
                              eval_metric='accuracy').fit(
    train_data,
    time_limit=3600,  # 1小时
    presets='best_quality'  # 或 'medium_quality', 'optimize_for_deployment'
)

# 预测
predictions = predictor.predict(test_data)

# 查看排行榜
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)
```


### 3.2 AutoGluon 的堆叠策略


| 层 | 包含的模型 | 作用 |
| --- | --- | --- |
| Layer 1 | 随机森林、GBM、KNN、神经网络、XT等 | 基础预测 |
| Layer 2 | 使用Layer 1输出作为特征的GBM | 学习基础模型的组合 |
| Layer 3 | 加权集成 | 最终预测 |


## 四、H2O AutoML


H2O AutoML 是 H2O.ai 开发的分布式 AutoML 平台，支持大规模数据处理。


```
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# 导入数据
train = h2o.import_file("train.csv")
test = h2o.import_file("test.csv")

# 设置特征和目标
x = train.columns
y = "target"
x.remove(y)

# 运行AutoML
aml = H2OAutoML(
    max_models=20,
    seed=42,
    max_runtime_secs=600,
    sort_metric="AUC"       # 分类用AUC，回归用RMSE
)
aml.train(x=x, y=y, training_frame=train)

# 查看排行榜
lb = aml.leaderboard
print(lb.head(10))

# 最佳模型预测
preds = aml.leader.predict(test)
h2o.shutdown()
```


## 五、各框架对比


| 框架 | 开发商 | 擅长领域 | 分布式 | 学习曲线 |
| --- | --- | --- | --- | --- |
| AutoSklearn | 弗莱堡大学 | 表格数据分类/回归 | 有限 | 中等 |
| AutoGluon | Amazon | 表格/文本/图像 | 支持 | 简单 |
| H2O AutoML | H2O.ai | 大规模表格数据 | 原生支持 | 中等 |
| Google Vertex AI | Google | 全平台 | 云端原生 | 简单 |
| FLAML | Microsoft | 高效低成本 | 有限 | 简单 |


## 六、何时使用 AutoML vs 手工调参


| 场景 | 推荐方法 | 原因 |
| --- | --- | --- |
| 快速原型验证 | AutoML | 几分钟出baseline |
| 表格数据竞赛 | AutoML + 手工优化 | AutoML先出80%方案，手工提升最后20% |
| 生产环境部署 | 手工为主 | 需要可控性、可解释性 |
| 非表格数据（NLP/CV） | 手工 + 超参数优化 | 需要领域知识和预训练模型 |
| 大规模数据 | H2O / 分布式框架 | 需要分布式计算能力 |
| 资源受限 | AutoML | 自动选择轻量模型 |


> **Important:** **AutoML的局限性：**
>
> - 无法处理需要领域知识的特征工程（如时间序列的滞后特征）
> - 对非标准问题（多任务学习、强化学习等）支持有限
> - 模型可能不是最优的——手工精心调参通常能超过AutoML
> - 可解释性差：自动选择的模型可能是黑盒集成
> - 不处理数据质量：数据收集、清洗、标注仍需人工


## 七、FLAML — 微软的高效 AutoML


FLAML（Fast Lightweight AutoML）由微软开发，专注于**低成本高效**的超参数优化。


```
from flaml import AutoML
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

automl = AutoML()
automl.fit(X_train, y_train,
           task="classification",
           time_budget=60,          # 60秒时间预算
           metric="accuracy",
           estimator_list=["lgbm", "rf", "xgboost", "extra_tree"])

print(f"最优估计器: {automl.best_estimator}")
print(f"最优配置: {automl.best_config}")
print(f"测试准确率: {automl.score(X_test, y_test):.4f}")
```


> **Note:** **FLAML的特点：**
> 成本驱动型优化（指定时间和预算），使用创新的搜索策略而非传统贝叶斯优化，在有限预算下往往表现最好。


## 总结


- AutoML自动化了ML流程的数据预处理、特征工程、模型选择和超参数优化
- AutoSklearn使用贝叶斯优化+元学习，适合传统ML
- AutoGluon开箱即用，多层堆叠集成提供高精度
- H2O AutoML原生支持分布式，适合大规模数据
- FLAML在有限预算下效率最高
- AutoML适合快速原型和表格数据，但无法替代领域知识和手工优化
- 实际工作中建议AutoML出baseline，手工优化做最终提升


<!-- Converted from: 02_AutoML框架.html -->
