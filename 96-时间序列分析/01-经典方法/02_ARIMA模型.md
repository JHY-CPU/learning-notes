# ARIMA模型


## 一、AR(p) 自回归模型


自回归模型（AutoRegressive）的思想是：当前值可以表示为其过去 p 个值的线性组合加上白噪声。


$$
AR(p) 模型：
                Xt = c + φ1Xt-1 + φ2Xt-2 + ... + φpXt-p + εt
$$


其中 c 为常数项，φ~1~, ..., φ~p~ 为自回归系数，ε~t~ ~ WN(0, σ²) 为白噪声。


### AR(1) 模型的性质


- 均值：E(X
   ~t~
   ) = c / (1 - φ
   ~1~
   )
- 平稳条件：|φ
   ~1~
   | < 1
- ACF：拖尾，按指数衰减，ρ(k) = φ
   ~1~
   ^k^
- PACF：1阶截尾，即 k > 1 时 PACF(k) = 0


> **Note:** **AR模型的核心思想：**
> 类似于线性回归，但回归变量不是其他特征，而是序列自身的滞后值。这使得AR模型能够捕捉序列的时间依赖结构。


## 二、MA(q) 移动平均模型


移动平均模型（Moving Average）将当前值表示为过去 q 个白噪声的线性组合。


$$
MA(q) 模型：
                Xt = μ + εt + θ1εt-1 + θ2εt-2 + ... + θqεt-q
$$


### MA(1) 模型的性质


- 均值：E(X
   ~t~
   ) = μ
- 方差：Var(X
   ~t~
   ) = σ²(1 + θ
   ~1~
   ²)
- 可逆条件：|θ
   ~1~
   | < 1
- ACF：1阶截尾，即 k > 1 时 ACF(k) = 0
- PACF：拖尾，按指数衰减


> **Important:** **注意：**
> 这里的"移动平均"与简单的滑动窗口平均不同。MA模型是对不可观测的随机误差项的加权平均，而不是对观测值的平均。


## 三、ARIMA(p,d,q) 模型


ARIMA（AutoRegressive Integrated Moving Average）将 AR 和 MA 模型与差分（Integrated）结合，可以处理非平稳时间序列。


$$
ARIMA(p,d,q) 模型：
                对原始序列做 d 阶差分后，得到平稳序列 Wt = ∇dXt，然后拟合 ARMA(p,q)：
                Wt = c + φ1Wt-1 + ... + φpWt-p + εt + θ1εt-1 + ... + θqεt-q
$$


| 参数 | 含义 | 确定方法 |
| --- | --- | --- |
| **p**（AR阶数） | 自回归项数 | PACF截尾阶数 |
| **d**（差分阶数） | 使序列平稳的差分次数 | ADF检验确定 |
| **q**（MA阶数） | 移动平均项数 | ACF截尾阶数 |


### ARIMA建模流程（Box-Jenkins方法）


1. **识别阶段：**
   检验平稳性，确定 d；通过ACF/PACF确定 p 和 q 的候选值
2. **估计阶段：**
   用最大似然估计或最小二乘估计参数
3. **诊断阶段：**
   检验残差是否为白噪声（Ljung-Box检验）
4. **预测阶段：**
   使用拟合好的模型进行预测


## 四、模型选择：AIC 与 BIC 准则


当多个 (p,q) 组合都可以通过残差检验时，使用信息准则来选择最优模型。


$$
AIC（赤池信息准则）：AIC = -2ln(L) + 2k
BIC（贝叶斯信息准则）：BIC = -2ln(L) + k·ln(n)
$$


其中 L 为似然函数值，k 为参数个数，n 为样本量。AIC和BIC越小越好。


| 准则 | 特点 | 适用场景 |
| --- | --- | --- |
| AIC | 倾向于选择更复杂的模型 | 预测导向，小样本 |
| BIC | 惩罚项更重，倾向简单模型 | 模型解释导向，大样本 |


> **Note:** **实践建议：**
> 通常先用ACF/PACF图缩小候选范围，然后在候选范围内用AIC/BIC做最终选择。可以使用statsmodels的auto_arima自动搜索最优参数。


## 五、SARIMA — 季节性 ARIMA


当时间序列存在季节性模式时，使用SARIMA（Seasonal ARIMA）模型，在ARIMA的基础上增加季节性参数。


$$
SARIMA(p,d,q)(P,D,Q)s
                其中 (p,d,q) 为非季节性参数，(P,D,Q) 为季节性参数，s 为季节周期
$$


| 参数 | 含义 | 示例（月度数据，s=12） |
| --- | --- | --- |
| P | 季节性AR阶数 | 依赖前1年、2年的同期值 |
| D | 季节性差分阶数 | X~t~ - X~t-12~ |
| Q | 季节性MA阶数 | 依赖前1年、2年的同期误差 |
| s | 季节周期 | 月度=12，季度=4，日度=7或365 |


## 六、Python 实战：ARIMA 建模与预测


> **Example:** ### 示例：航空乘客数据 SARIMA 建模
>
>
> ```
> import numpy as np
> import pandas as pd
> import matplotlib.pyplot as plt
> from statsmodels.tsa.arima.model import ARIMA
> from statsmodels.tsa.statespace.sarimax import SARIMAX
> from statsmodels.tsa.stattools import adfuller
> from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
> from statsmodels.stats.diagnostic import acorr_ljungbox
> import warnings
> warnings.filterwarnings('ignore')
>
> # 1. 加载航空乘客数据
> from statsmodels.datasets import co2
> data = co2.load().data
> data = data.resample('M').mean().ffill()
> print(f"数据范围: {data.index[0]} 到 {data.index[-1]}")
> print(f"样本量: {len(data)}")
>
> # 2. 可视化原始数据
> fig, axes = plt.subplots(2, 2, figsize=(14, 10))
> axes[0, 0].plot(data)
> axes[0, 0].set_title("原始CO2浓度数据")
>
> # 3. 平稳性检验与差分
> def check_stationarity(series, name):
>     result = adfuller(series.dropna())
>     print(f"{name}: ADF统计量={result[0]:.4f}, p值={result[1]:.4f}")
>     return result[1] < 0.05
>
> print("\n=== 平稳性检验 ===")
> is_stationary = check_stationarity(data, "原始序列")
> if not is_stationary:
>     diff1 = data.diff().dropna()
>     is_stationary = check_stationarity(diff1, "一阶差分")
>
> # 4. ACF/PACF分析（差分后）
> diff_data = data.diff(12).diff().dropna()  # 季节差分 + 普通差分
> plot_acf(diff_data, lags=40, ax=axes[0, 1], title="差分后 ACF")
> plot_pacf(diff_data, lags=40, ax=axes[1, 0], title="差分后 PACF")
>
> # 5. SARIMA建模
> train = data[:'2001']
> test = data['2002':]
>
> model = SARIMAX(train,
>                 order=(1, 1, 1),
>                 seasonal_order=(1, 1, 1, 12),
>                 enforce_stationarity=False,
>                 enforce_invertibility=False)
> results = model.fit(disp=False)
> print("\n=== SARIMA模型摘要 ===")
> print(f"AIC: {results.aic:.2f}")
> print(f"BIC: {results.bic:.2f}")
>
> # 6. 残差诊断
> residuals = results.resid
> lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
> print("\n=== 残差Ljung-Box检验 ===")
> print(lb_test)
>
> axes[1, 1].plot(residuals)
> axes[1, 1].set_title("模型残差")
> plt.tight_layout()
> plt.savefig("sarima_analysis.png", dpi=150)
> plt.show()
>
> # 7. 预测
> forecast = results.get_forecast(steps=len(test))
> pred_mean = forecast.predicted_mean
> pred_ci = forecast.conf_int()
>
> plt.figure(figsize=(12, 5))
> plt.plot(train[-120:], label='训练数据')
> plt.plot(test, label='实际值', color='green')
> plt.plot(pred_mean, label='预测值', color='red', linestyle='--')
> plt.fill_between(pred_ci.index,
>                  pred_ci.iloc[:, 0],
>                  pred_ci.iloc[:, 1], alpha=0.2, color='red')
> plt.legend()
> plt.title("SARIMA预测结果")
> plt.tight_layout()
> plt.savefig("sarima_forecast.png", dpi=150)
> plt.show()
>
> # 8. 评估
> from sklearn.metrics import mean_absolute_error, mean_squared_error
> mae = mean_absolute_error(test, pred_mean)
> rmse = np.sqrt(mean_squared_error(test, pred_mean))
> print(f"\n=== 预测评估 ===")
> print(f"MAE: {mae:.4f}")
> print(f"RMSE: {rmse:.4f}")
> ```


## 七、auto_arima 自动选参


手动选择 (p,d,q) 和 (P,D,Q,s) 参数较为繁琐，pmdarima 库提供了 auto_arima 函数自动搜索最优参数。


```
import pmdarima as pm

# 自动搜索最优SARIMA参数
auto_model = pm.auto_arima(
    train,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None,           # 自动确定差分阶数
    seasonal=True,
    m=12,             # 季节周期
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    D=None,           # 自动确定季节差分阶数
    information_criterion='aic',
    stepwise=True,
    trace=True,       # 打印搜索过程
    error_action='ignore',
    suppress_warnings=True
)

print(f"最优模型: {auto_model.order}{auto_model.seasonal_order}")
print(auto_model.summary())
```


## 总结


- AR(p)利用历史值的线性组合预测，PACF截尾确定阶数
- MA(q)利用历史误差的线性组合预测，ACF截尾确定阶数
- ARIMA(p,d,q)通过差分处理非平稳序列
- AIC和BIC用于模型选择，BIC偏好更简单的模型
- SARIMA在ARIMA基础上加入季节性参数，处理周期性数据
- auto_arima可以自动搜索最优参数组合
- 建模后必须检验残差是否为白噪声


<!-- Converted from: 02_ARIMA模型.html -->
