# 指数平滑与Prophet


## 一、简单指数平滑（Simple Exponential Smoothing）


简单指数平滑适用于**无趋势、无季节性**的时间序列。其核心思想是对历史观测值进行加权平均，越近的观测值权重越大。


$$
递推公式：
                ˆXt+1 = αXt + (1 - α)ˆXt
                展开形式：ˆXt+1 = αXt + α(1-α)Xt-1 + α(1-α)²Xt-2 + ...
$$


其中 α ∈ (0, 1) 为平滑参数。α 越大，近期观测值权重越大，模型对变化越敏感；α 越小，模型越平滑。


| α 值 | 行为特征 | 适用场景 |
| --- | --- | --- |
| α → 0 | 几乎不更新，预测值接近历史均值 | 非常平稳的序列 |
| α ∈ [0.1, 0.3] | 较平滑，抑制噪声 | 缓慢变化的序列 |
| α ∈ [0.3, 0.7] | 平衡灵敏度和平滑性 | 一般场景 |
| α → 1 | 几乎只看最近一个值 | 快速变化的序列 |


## 二、Holt 线性趋势法


Holt 方法在简单指数平滑的基础上增加了**趋势分量**，可以处理有线性趋势的时间序列。引入了两个平滑参数。


$$
Holt双参数指数平滑：
                水平：lt = αyt + (1-α)(lt-1 + bt-1)
                趋势：bt = β(lt - lt-1) + (1-β)bt-1
                预测：ˆXt+h = lt + h · bt
$$


其中 α 为水平平滑参数，β 为趋势平滑参数，h 为预测步长。


> **Note:** **Holt阻尼趋势法：**
> 当趋势不宜长期外推时，引入阻尼参数 φ ∈ (0,1)，使趋势逐渐减弱：
>
>
> ˆX
> ~t+h~
> = l
> ~t~
> + (φ + φ² + ... + φ
> ^h^
> )b
> ~t~
> ，预测值最终趋于水平。


## 三、Holt-Winters 方法


Holt-Winters 在Holt方法的基础上增加了**季节性分量**，是指数平滑族中最完整的方法，包含三个平滑参数。


### 3.1 加法季节模型


$$
水平：lt = α(yt - st-m) + (1-α)(lt-1 + bt-1)
                趋势：bt = β(lt - lt-1) + (1-β)bt-1
                季节：st = γ(yt - lt-1 - bt-1) + (1-γ)st-m
                预测：ˆXt+h = lt + h·bt + st+h-m
$$


### 3.2 乘法季节模型


$$
水平：lt = α(yt / st-m) + (1-α)(lt-1 + bt-1)
                趋势：bt = β(lt - lt-1) + (1-β)bt-1
                季节：st = γ(yt / (lt-1 + bt-1)) + (1-γ)st-m
                预测：ˆXt+h = (lt + h·bt) × st+h-m
$$


| 参数 | 范围 | 控制的分量 |
| --- | --- | --- |
| α | [0, 1] | 水平（level）的更新速度 |
| β | [0, 1] | 趋势（trend）的更新速度 |
| γ | [0, 1] | 季节（seasonal）的更新速度 |
| m | 正整数 | 季节周期长度 |


## 四、Facebook Prophet


Prophet 是 Facebook（Meta）于 2017 年开源的时间序列预测框架，特别适用于具有强季节性和假日效应的业务数据。


### 4.1 Prophet 的模型结构


$$
Prophet 加法模型：
                y(t) = g(t) + s(t) + h(t) + εt
                g(t)：趋势函数（分段线性或Logistic增长）
                s(t)：季节性函数（傅里叶级数）
                h(t)：假日效应
                εt：随机误差
$$


### 4.2 分段线性趋势


$$
g(t) = (k + a(t)Tδ)t + (m + a(t)Tγ)
                k：基础增长率
                δ：增长率变化量向量（在变点处）
                a(t)：指示向量，标记 t 是否超过某个变点
$$


### 4.3 傅里叶级数建模季节性


$$
s(t) = ∑n=1N [ancos(2πnt/P) + bnsin(2πnt/P)]
$$


其中 P 为周期（年=365.25，周=7），N 控制傅里叶项数（N越大越灵活，但也越容易过拟合）。


### 4.4 Prophet vs ARIMA 对比


| 特性 | ARIMA | Prophet |
| --- | --- | --- |
| 数据要求 | 等间距、无缺失 | 可处理缺失和不等间距 |
| 季节性处理 | 需指定周期参数 | 自动检测多种季节性 |
| 假日效应 | 不支持 | 内置支持 |
| 趋势变化点 | 不支持 | 自动检测变点 |
| 可解释性 | 参数级 | 组件级可视化 |
| 适用场景 | 统计建模、短序列 | 业务数据、长序列 |


## 五、Python 实战：Prophet 预测


> **Example:** ### 示例：使用 Prophet 预测航空乘客数据
>
>
> ```
> import pandas as pd
> import numpy as np
> import matplotlib.pyplot as plt
> from prophet import Prophet
>
> # 1. 准备数据（Prophet要求列名为 ds 和 y）
> from statsmodels.datasets import co2
> raw = co2.load().data
> df = pd.DataFrame({
>     'ds': raw.index,
>     'y': raw.values.flatten()
> }).resample('M', on='ds').mean().reset_index()
> df = df.dropna()
>
> # 划分训练测试集
> train = df[df['ds'] < '2002-01-01']
> test = df[df['ds'] >= '2002-01-01']
> print(f"训练集: {len(train)} 条, 测试集: {len(test)} 条")
>
> # 2. 创建并拟合Prophet模型
> model = Prophet(
>     growth='linear',           # 线性趋势
>     yearly_seasonality=True,   # 年季节性
>     weekly_seasonality=False,  # 关闭周季节性
>     daily_seasonality=False,   # 关闭日季节性
>     changepoint_prior_scale=0.05,  # 趋势变化点的灵活性
>     seasonality_prior_scale=10,    # 季节性的灵活性
>     seasonality_mode='additive'    # 加法季节性
> )
>
> # 添加自定义季节性（可选）
> model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
>
> model.fit(train)
>
> # 3. 创建未来日期并预测
> future = model.make_future_dataframe(periods=len(test), freq='M')
> forecast = model.predict(future)
>
> # 4. 可视化预测结果
> fig1 = model.plot(forecast)
> plt.title("Prophet预测结果")
> plt.tight_layout()
> plt.savefig("prophet_forecast.png", dpi=150)
> plt.show()
>
> # 5. 可视化各组件（趋势+季节性）
> fig2 = model.plot_components(forecast)
> plt.savefig("prophet_components.png", dpi=150)
> plt.show()
>
> # 6. 评估
> from sklearn.metrics import mean_absolute_error, mean_squared_error
> pred = forecast[forecast['ds'] >= '2002-01-01'][['ds', 'yhat']].head(len(test))
> merged = test.merge(pred, on='ds')
> mae = mean_absolute_error(merged['y'], merged['yhat'])
> rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
> print(f"\n=== Prophet预测评估 ===")
> print(f"MAE: {mae:.4f}")
> print(f"RMSE: {rmse:.4f}")
>
> # 7. 添加假日效应示例
> holidays = pd.DataFrame({
>     'holiday': ['earth_day', 'earth_day', 'earth_day'],
>     'ds': pd.to_datetime(['2000-04-22', '2001-04-22', '2002-04-22']),
>     'lower_window': -1,
>     'upper_window': 1,
> })
>
> model_with_holidays = Prophet(holidays=holidays)
> model_with_holidays.fit(train)
> print("假日效应已添加到模型中")
> ```


## 六、ETS 模型（Error-Trend-Seasonality）


ETS 模型是指数平滑方法的统一状态空间表示，由 Hyndman 等人提出。每种指数平滑方法都可以表示为一个 ETS 模型。


| ETS名称 | 趋势 | 季节性 | 对应方法 |
| --- | --- | --- | --- |
| ETS(A,N,N) | 无 | 无 | 简单指数平滑 |
| ETS(A,A,N) | 加法 | 无 | Holt线性趋势 |
| ETS(A,Ad,N) | 阻尼加法 | 无 | Holt阻尼趋势 |
| ETS(A,A,A) | 加法 | 加法 | Holt-Winters加法 |
| ETS(A,A,M) | 加法 | 乘法 | Holt-Winters乘法 |


```
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# 自动选择最优ETS模型
ets_model = ETSModel(train['y'], error='add', trend='add', seasonal='add',
                      seasonal_periods=12)
ets_result = ets_model.fit()
print(ets_result.summary())

# 预测
ets_forecast = ets_result.forecast(steps=len(test))
print(f"ETS预测完成，RMSE: {np.sqrt(mean_squared_error(test['y'], ets_forecast)):.4f}")
```


## 总结


- 简单指数平滑适合无趋势无季节性的平稳序列
- Holt方法处理线性趋势，引入阻尼可避免趋势无限外推
- Holt-Winters加入季节分量，加法/乘法取决于季节波动幅度
- Prophet集趋势、季节、假日于一体，适合业务时间序列预测
- Prophet的傅里叶级数可以灵活建模多种周期的季节性
- ETS模型统一了指数平滑族，便于模型选择
- 实际应用中建议同时尝试ARIMA和Prophet，选择效果更好的模型


<!-- Converted from: 03_指数平滑与Prophet.html -->
