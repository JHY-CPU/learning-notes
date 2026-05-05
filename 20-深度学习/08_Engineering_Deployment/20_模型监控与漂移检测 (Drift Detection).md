# 20_模型监控与漂移检测 (Drift Detection)

## 核心概念

- **模型漂移 (Model Drift)**：部署后的模型性能随时间逐渐下降的现象。根本原因是生产环境中的真实数据分布与训练数据分布发生了偏离。漂移是几乎所有生产系统都会面临的问题，持续监控是 MLOps 的核心能力。
- **数据漂移 (Data Drift)**：输入特征分布的变化。例如，用户的年龄分布从训练时的 18-30 岁变为部署后的 30-50 岁。检测方法包括统计检验（KS 检验、卡方检验）、分布距离计算（Wasserstein 距离、Jensen-Shannon 散度）和对抗验证。
- **概念漂移 (Concept Drift)**：输入与输出之间的映射关系发生了变化。例如，因为市场政策变化，过去被认为是"高信用"的用户现在可能变成了"低信用"。概念漂移通常更难检测，需要持续监控模型输出分布和业务指标。
- **标签漂移 (Label Drift)**：真实标签分布的变化。通常需要滞后获取人工标注或真实反馈才能检测（如贷款催收结果的滞后反馈）。这是最常见也最难实时检测的漂移类型。
- **漂移检测指标**：用于量化漂移程度的核心指标包括：PSI (Population Stability Index)、KL 散度、JS 散度、Wasserstein 距离、最大均值差异 (MMD)。其中 PSI 是金融风控领域的行业标准。
- **告警与自动回滚**：当漂移指标超过阈值时触发告警，并自动执行预设的应对策略：模型回滚到上一版本、触发重新训练流水线、人工介入审核、或将异常的流量切换到兜底规则引擎。

## 数学推导

**PSI (Population Stability Index)** 是最常用的特征分布漂移指标。将特征值分为 K 个桶（bin），比较生产分布 $P$ 与基准分布 $Q$ 在每个桶中的占比：

$$
\text{PSI} = \sum_{i=1}^{K} (P_i - Q_i) \cdot \ln\left(\frac{P_i}{Q_i}\right)
$$

- PSI < 0.1: 分布稳定
- 0.1 <= PSI < 0.2: 轻度漂移，需要关注
- PSI >= 0.2: 严重漂移，需要立即行动

**KS 检验 (Kolmogorov-Smirnov Test)** 用于检测连续特征分布的差异：

$$
D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|
$$

其中 $F_{1,n}$ 和 $F_{2,m}$ 是两个样本的经验累积分布函数。$D$ 值越大，分布差异越显著。

**对抗验证 (Adversarial Validation)** 是一种数据驱动的漂移检测方法。训练一个二分类器来区分训练集样本和生产集样本：

$$
\mathcal{L}_{\text{adv}} = -\frac{1}{N}\sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$

其中 $y_i=1$ 表示训练集样本，$y_i=0$ 表示生产集样本。如果分类器的 AUC > 0.8，表明两个分布存在显著差异，即发生了数据漂移。

## 直观理解

- **模型漂移 = 地图过期**：训练好的模型就像一张地图。城市在发展（数据分布变化），昨天的地图在今天可能就不准确了。漂移检测就是定期检查地图与实际情况的差异，当误差太大时就更新地图（重新训练模型）。
- **数据漂移 vs 概念漂移**：数据漂移是"输入范围变了"——以前只见过 10-30 岁的用户，现在来了 50 岁的；概念漂移是"规则变了"——以前穿拖鞋的人不允许进餐厅（分类为"非顾客"），现在政策变了穿拖鞋也可以进。两者往往同时发生。
- **最佳实践**：先监控模型输出的 PSI（最容易实施），再监控关键特征的分布；建立分级告警制度（黄牌警告、红牌行动）；每次模型更新后更新基准分布。
- **常见陷阱**：不要使用太小的窗口检测漂移（短期波动不是漂移），建议使用 7 天或 30 天滚动窗口；季节性业务（如双十一流量暴增）需要用同期历史数据作为对比基准。

## 代码示例

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta

# ========== 1. PSI 计算 ==========
def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """计算 Population Stability Index"""
    # 将数据分桶
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # 计算两个分布在每个桶中的占比
    expected_percents = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bin_edges)[0] / len(actual)

    # 处理零值（避免除零错误）
    expected_percents = np.clip(expected_percents, 1e-6, 1)
    actual_percents = np.clip(actual_percents, 1e-6, 1)

    # 计算 PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi


# ========== 2. 多特征漂移检测器 ==========
class DriftDetector:
    """检测多个特征的数据漂移"""
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.2):
        """
        Args:
            reference_data: 训练集数据（基准分布）
            threshold: PSI 阈值，超过即告警
        """
        self.reference = reference_data
        self.threshold = threshold
        self.feature_bounds = {}
        self._prepare_reference()

    def _prepare_reference(self):
        """为每个连续特征准备基准统计"""
        for col in self.reference.select_dtypes(include=[np.number]).columns:
            self.feature_bounds[col] = {
                "min": self.reference[col].min(),
                "max": self.reference[col].max(),
                "mean": self.reference[col].mean(),
                "std": self.reference[col].std(),
            }

    def detect_drift(self, production_data: pd.DataFrame) -> Dict[str, float]:
        """检测生产数据的漂移情况"""
        results = {}
        alerts = []

        for col in self.reference.select_dtypes(include=[np.number]).columns:
            if col not in production_data.columns:
                continue

            ref_values = self.reference[col].values
            prod_values = production_data[col].values

            # 使用 KS 检验和 PSI
            ks_stat, ks_pval = stats.ks_2samp(ref_values, prod_values)
            psi = calculate_psi(ref_values, prod_values)

            results[col] = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "ref_mean": self.feature_bounds[col]["mean"],
                "prod_mean": np.mean(prod_values),
            }

            if psi > self.threshold:
                alerts.append({
                    "feature": col,
                    "psi": psi,
                    "message": f"特征 '{col}' 发生严重漂移 (PSI={psi:.3f})"
                })

        return {"metrics": results, "alerts": alerts}

    def generate_report(self, production_data: pd.DataFrame) -> str:
        """生成漂移检测报告"""
        result = self.detect_drift(production_data)
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_features": len(result["metrics"]),
            "drifted_features": len(result["alerts"]),
            "alerts": result["alerts"],
            "details": result["metrics"]
        }
        return json.dumps(report, indent=2, default=str)


# ========== 3. 对抗验证 ==========
def adversarial_drift_detection(
    train_features: pd.DataFrame,
    prod_features: pd.DataFrame,
    n_estimators: int = 100
) -> float:
    """通过对抗验证检测数据漂移"""
    # 构建数据集：训练集标 1，生产集标 0
    X = pd.concat([train_features, prod_features], ignore_index=True)
    y = np.concatenate([
        np.ones(len(train_features)),
        np.zeros(len(prod_features))
    ])

    # 训练分类器
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X, y)

    # 交叉验证 AUC
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    mean_auc = np.mean(scores)

    return mean_auc  # AUC > 0.8 表示存在显著漂移


# ========== 4. 生产级监控系统（简化）===========
class ModelMonitor:
    """模型监控系统，集成漂移检测和告警"""
    def __init__(self, model_id: str, feature_names: List[str],
                 ref_data: pd.DataFrame, psi_threshold: float = 0.2):
        self.model_id = model_id
        self.detector = DriftDetector(ref_data, psi_threshold)
        self.feature_names = feature_names
        self.metrics_history = []
        self.alert_callbacks = []

    def register_alert_callback(self, callback):
        """注册告警回调函数（如发送 Slack 消息）"""
        self.alert_callbacks.append(callback)

    def monitor_batch(self, production_data: pd.DataFrame,
                      predictions: np.ndarray) -> Dict:
        """监控一批生产数据"""
        # 数据漂移检测
        drift_result = self.detector.detect_drift(production_data)

        # 对抗验证（可选）
        adv_auc = adversarial_drift_detection(
            self.detector.reference[self.feature_names],
            production_data[self.feature_names]
        )

        # 输出分布监控
        output_stats = {
            "pred_mean": float(np.mean(predictions)),
            "pred_std": float(np.std(predictions)),
            "pred_min": float(np.min(predictions)),
            "pred_max": float(np.max(predictions)),
        }

        # 记录历史
        entry = {
            "timestamp": datetime.now(),
            "n_samples": len(production_data),
            "drift_alerts": len(drift_result["alerts"]),
            "adversarial_auc": adv_auc,
            "output_stats": output_stats,
        }
        self.metrics_history.append(entry)

        # 触发告警
        if drift_result["alerts"]:
            for alert in drift_result["alerts"]:
                print(f"[ALERT] {alert['message']}")
                for callback in self.alert_callbacks:
                    callback(self.model_id, alert)

        # 告警等级判断
        severity = "normal"
        if len(drift_result["alerts"]) > 0:
            severity = "warning"
        if adv_auc > 0.8:
            severity = "critical"
        if len(drift_result["alerts"]) >= 3:
            severity = "critical"

        return {
            "severity": severity,
            "drift_result": drift_result,
            "adversarial_auc": adv_auc,
            "output_stats": output_stats,
        }


# ========== 5. 示例：模拟漂移检测 ==========
def demo_drift_detection():
    print("=" * 60)
    print("模型漂移检测演示")
    print("=" * 60)

    # 生成基准数据（训练集分布）
    np.random.seed(42)
    n_train = 10000
    train_data = pd.DataFrame({
        "age": np.random.normal(30, 10, n_train).clip(18, 80),
        "income": np.random.lognormal(4, 0.5, n_train),
        "credit_score": np.random.normal(700, 100, n_train).clip(300, 850),
        "num_transactions": np.random.poisson(20, n_train),
    })

    # 生成正常生产数据（无漂移）
    normal_prod = pd.DataFrame({
        "age": np.random.normal(30, 10, 1000).clip(18, 80),
        "income": np.random.lognormal(4, 0.5, 1000),
        "credit_score": np.random.normal(700, 100, 1000).clip(300, 850),
        "num_transactions": np.random.poisson(20, 1000),
    })

    # 生成漂移生产数据
    drift_prod = pd.DataFrame({
        "age": np.random.normal(45, 15, 1000).clip(18, 80),  # 年龄分布偏移
        "income": np.random.lognormal(4.5, 0.7, 1000),       # 收入分布偏移
        "credit_score": np.random.normal(650, 120, 1000).clip(300, 850),
        "num_transactions": np.random.poisson(15, 1000),     # 交易次数偏移
    })

    # 创建检测器
    detector = DriftDetector(train_data, threshold=0.1)

    print("\n正常生产数据检测：")
    normal_result = detector.detect_drift(normal_prod)
    print(f"  告警数: {len(normal_result['alerts'])}")
    for col, metrics in normal_result["metrics"].items():
        print(f"  {col}: PSI={metrics['psi']:.4f}")

    print("\n漂移生产数据检测：")
    drift_result = detector.detect_drift(drift_prod)
    print(f"  告警数: {len(drift_result['alerts'])}")
    for alert in drift_result["alerts"]:
        print(f"  [ALERT] {alert['message']}")

    print("\n对抗验证结果：")
    normal_auc = adversarial_drift_detection(train_data, normal_prod)
    drift_auc = adversarial_drift_detection(train_data, drift_prod)
    print(f"  无漂移数据 AUC: {normal_auc:.3f}")
    print(f"  漂移数据 AUC:   {drift_auc:.3f}")

# if __name__ == "__main__":
#     demo_drift_detection()


# ========== 6. 概念漂移检测 ==========
class ConceptDriftDetector:
    """检测概念漂移（输出-标签映射关系变化）"""
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.accuracy_history = []

    def update(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """更新准确率历史（需要真实标签）"""
        accuracy = np.mean(predictions == ground_truth)
        self.accuracy_history.append(accuracy)

        # 只保留最近 window_size 个样本的准确率
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history = self.accuracy_history[-self.window_size:]

    def detect(self) -> Dict:
        """检测是否存在概念漂移"""
        if len(self.accuracy_history) < 100:
            return {"drift_detected": False, "reason": "样本不足"}

        # 将历史分为前一半和后一半
        mid = len(self.accuracy_history) // 2
        early_avg = np.mean(self.accuracy_history[:mid])
        recent_avg = np.mean(self.accuracy_history[mid:])

        # 如果最近准确率显著下降，则认为发生概念漂移
        drop = early_avg - recent_avg
        drift_detected = drop > 0.05  # 5% 阈值

        return {
            "drift_detected": drift_detected,
            "early_accuracy": early_avg,
            "recent_accuracy": recent_avg,
            "accuracy_drop": drop,
            "samples_seen": len(self.accuracy_history),
        }
```

## 深度学习关联

- **MLOps 持续监控的核心组件**：模型漂移检测是 MLOps 闭环的关键环节。典型的生产系统监控方案包括：日志聚合（ELK Stack/Loki）保存推理请求和响应 -> Prometheus 采集 PSI/KS 等指标 -> Grafana 可视化仪表盘 -> 超过阈值触发 Alertmanager -> 自动调用训练流水线生成新模型 -> 通过 A/B 测试验证后自动部署。这一完整闭环是 MLOps 成熟度从 Level 2 提升到 Level 3 的标志。
- **AI 系统的可观测性标准 OTel**：OpenTelemetry 正在制定 AI 系统的可观测性标准，模型漂移检测指标（特征分布距离、模型置信度分布、数据完整性评分）将被纳入统一的遥测数据模型。这为跨平台、跨团队的模型监控标准化奠定了基础。
- **数据湖与特征管道集成**：漂移检测通常与数据湖（Delta Lake/Iceberg）集成——生产环境中的推理数据被自动流入数据湖的时间分区表中，由调度任务定时（每小时/每天）计算漂移指标并与基准分布比较。检测结果写回 MLflow 或专门的模型监控平台（如 WhyLabs、Evidently AI、SageMaker Model Monitor）。
