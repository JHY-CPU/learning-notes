# MLOps核心实践 - 模型监控与MLOps


## 1. MLOps定义与价值


MLOps（Machine Learning Operations）是机器学习的DevOps，旨在将ML模型从实验阶段可靠、高效地部署到生产环境，并持续维护和迭代。


| 维度 | 传统ML开发 | MLOps实践 |
| --- | --- | --- |
| 实验管理 | Jupyter Notebook、手动记录 | 自动追踪超参、指标、代码版本 |
| 模型部署 | 手动导出、手动部署 | CI/CD自动部署、容器化 |
| 数据管理 | 手动处理，无版本控制 | 数据版本化、血缘追踪 |
| 模型监控 | 缺乏监控，问题发现滞后 | 实时监控、漂移检测、自动告警 |
| 团队协作 | 各自为战，难以复现 | 标准化流程、可复现实验 |


> **Note:** MLOps成熟度等级：
>
>
> Level 0（手动）：手动管理一切
>
>
> Level 1（ML Pipeline自动化）：自动化训练管道
>
>
> Level 2（CI/CD Pipeline）：自动化测试和部署
>
>
> Level 3（全自动MLOps）：监控→触发→训练→部署全自动


## 2. ML生命周期


> **Example:** **完整ML生命周期：**
>
>
> 数据收集 → 数据验证 → 特征工程 → 模型训练 → 模型评估 → 模型注册 → 模型部署 → 模型监控 → (数据漂移) → 回到数据收集


| 阶段 | 核心活动 | 关键工具 |
| --- | --- | --- |
| 数据管理 | 采集、清洗、版本化、验证 | DVC、Delta Lake、Great Expectations |
| 特征工程 | 特征定义、转换、存储 | Feast、Tecton |
| 模型训练 | 实验管理、超参搜索、分布式训练 | MLflow、W&B、Neptune |
| 模型评估 | 离线评估、A/B测试、公平性检查 | MLflow、自定义评估框架 |
| 模型部署 | 容器化、服务化、灰度发布 | Docker、K8s、Seldon Core |
| 模型监控 | 漂移检测、性能监控、告警 | evidently、Prometheus、Grafana |


## 3. MLflow实验管理


MLflow是最流行的开源ML生命周期管理平台，包含四个核心组件：


- **MLflow Tracking：**
   记录和查询实验（参数、指标、模型、代码）
- **MLflow Projects：**
   可复现的代码运行环境
- **MLflow Models：**
   多种格式的模型打包和部署
- **Model Registry：**
   模型版本管理、阶段转换（Staging→Production→Archived）


```
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 设置实验
mlflow.set_experiment("iris-classification")

# 准备数据
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 开始实验跟踪
with mlflow.start_run(run_name="rf_baseline") as run:
    # 记录超参数
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)

    # 训练模型
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 预测与评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 记录指标
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # 记录模型（自动记录依赖和签名）
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="IrisClassifier"  # 注册到模型注册中心
    )

    print(f"实验Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

# 查询所有实验Runs
runs = mlflow.search_runs(experiment_ids=["0"])
print(runs[['run_id', 'metrics.accuracy', 'metrics.f1_score']])
```


## 4. 模型注册中心


| 模型阶段 | 含义 | 操作 |
| --- | --- | --- |
| None | 刚注册的模型 | 初始状态 |
| Staging | 测试/灰度验证阶段 | 通过CI测试后转入 |
| Production | 生产环境部署 | 通过验收后转入 |
| Archived | 已归档的旧版本 | 新版本上线后转移旧版本 |


```
# 通过MLflow API管理模型版本
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 获取模型最新版本
model_name = "IrisClassifier"
latest_versions = client.get_latest_versions(model_name, stages=["None"])

for version in latest_versions:
    print(f"Version: {version.version}, Stage: {version.current_stage}")

# 将模型转到Staging阶段
client.transition_model_version_stage(
    name=model_name,
    version=latest_versions[0].version,
    stage="Staging"
)

# 添加模型描述
client.update_model_version(
    name=model_name,
    version=latest_versions[0].version,
    description="随机森林基线模型，accuracy=0.97"
)
```


## 5. CI/CD for ML


### ML的CI/CD与传统软件的区别


| 维度 | 传统软件CI/CD | ML CI/CD |
| --- | --- | --- |
| 测试内容 | 代码逻辑正确性 | 代码正确性 + 数据质量 + 模型性能 |
| 通过标准 | 测试用例全部通过 | 模型指标达标 + 无数据异常 |
| 版本控制 | 代码版本（Git） | 代码 + 数据 + 模型 + 配置版本 |
| 回滚机制 | 代码回滚 | 模型版本回滚 |
| 环境一致性 | Docker镜像 | Docker + 模型运行时 + GPU驱动 |


### ML测试金字塔


1. **数据测试：**
   数据Schema验证、数据质量检查、数据量检查
2. **特征测试：**
   特征计算逻辑正确性、特征值范围检查
3. **模型测试：**
   指标必须不低于基线、公平性检查、推理延迟测试
4. **集成测试：**
   端到端推理链路测试
5. **线上验证：**
   A/B测试、金丝雀发布


> **Important:** 模型回滚是ML系统的关键能力。必须保留历史模型的完整信息（代码版本、数据版本、超参数、训练环境），以便在新模型出现问题时快速回退。


## 总结


- MLOps是机器学习的DevOps，目标是可靠、高效地将模型部署到生产并持续维护
- ML生命周期包含数据管理、特征工程、模型训练、评估、部署和监控六个阶段
- MLflow是最流行的开源MLOps平台，提供Tracking、Projects、Models、Registry四大组件
- 模型注册中心管理模型的生命周期：None→Staging→Production→Archived
- ML的CI/CD需要额外测试数据质量和模型性能，而不仅是代码正确性
- 模型版本管理和回滚能力是生产ML系统的基础要求


<!-- Converted from: 01_MLOps核心实践.html -->
