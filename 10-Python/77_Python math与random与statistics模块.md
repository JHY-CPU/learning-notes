# Python math与random与statistics模块


## 🔢 Python math / random / statistics 模块


math 数学函数与常量、random 随机数生成、statistics 基础统计计算。三个标准库模块覆盖科学计算与数据处理基础需求。


## math 模块: 常量


```
// ========== math 常量 ==========
import math

# 数学常量:
print(math.pi)       # 3.141592653589793
print(math.e)        # 2.718281828459045
print(math.tau)      # 6.283185307179586 (2π)
print(math.inf)      # inf (无穷大)
print(math.nan)      # nan (非数字)

# 无穷大判断:
x = float('inf')
print(math.isinf(x))     # True
print(math.isnan(math.nan))  # True
print(math.isfinite(42))     # True
```


## math 模块: 常用函数


```
// ========== 数学函数 ==========
import math

// ========== 取整 ==========
print(math.ceil(3.2))    # 4  向上取整
print(math.floor(3.8))   # 3  向下取整
print(math.trunc(3.8))   # 3  截断 (同 int())
print(round(3.5))        # 4  内置四舍五入 (银行家舍入)
print(round(4.5))        # 4  (Python 3 向偶取整)

// ========== 幂与对数 ==========
print(math.pow(2, 10))   # 1024.0  (2^10, 返回 float)
print(math.sqrt(16))     # 4.0    平方根
print(math.exp(2))       # 7.389... e^2
print(math.log(100))     # 4.605... 自然对数 ln(100)
print(math.log(100, 10)) # 2.0    以 10 为底
print(math.log2(8))      # 3.0    以 2 为底
print(math.log10(100))   # 2.0    以 10 为底

// ========== 三角函数 ==========
print(math.sin(math.pi/2))   # 1.0
print(math.cos(0))           # 1.0
print(math.tan(0))           # 0.0
print(math.asin(1))          # π/2 反正弦
print(math.acos(1))          # 0.0  反余弦
print(math.atan(1))          # π/4 反正切

# 角度转换:
print(math.degrees(math.pi))   # 180.0  弧度→度
print(math.radians(180))       # π      度→弧度

// ========== 其他 ==========
print(math.fabs(-3.5))     # 3.5  绝对值 (返回 float)
print(math.factorial(5))   # 120  阶乘 (5!)
print(math.gcd(12, 18))    # 6    最大公约数
print(math.lcm(12, 18))    # 36   最小公倍数 (3.9+)
print(math.comb(5, 2))     # 10   组合数 C(5,2)
print(math.perm(5, 2))     # 20   排列数 P(5,2)
print(math.fmod(10, 3))    # 1.0  取模 (float)
print(math.modf(3.14))     # (0.140..., 3.0) 小数和整数部分
```


## math 模块: 统计与距离


```
// ========== 统计与距离 ==========
import math

# 浮点数判断:
print(math.isclose(0.1 + 0.2, 0.3))  # True
# 比较两个浮点数是否近似相等
# 避免直接 == 比较浮点数

# 可调容差:
math.isclose(1.001, 1.002, rel_tol=0.01)  # True (1% 相对容差)
math.isclose(1.001, 1.002, abs_tol=0.01)  # True (0.01 绝对容差)

# 乘积和:
print(math.prod([1, 2, 3, 4]))     # 24  (乘积)
print(math.fsum([.1, .2, .3]))     # 0.6 (精确浮点和,用高精度累加)

# 欧几里得距离 (3.8+):
print(math.dist([0, 0], [3, 4]))   # 5.0

# 平方和 (3.8+):
print(math.hypot(3, 4))            # 5.0 sqrt(3² + 4²)

# 组合数 (3.8+):
# math.comb(5, 2) → 10
```


## random 模块: 基础随机


```
// ========== random 基础 ==========
import random

# [0, 1) 随机浮点数:
print(random.random())           # 0.374...

# [a, b] 随机整数 (包含两端):
print(random.randint(1, 10))     # 5

# [a, b) 随机整数 (不包含 b):
print(random.randrange(10))      # 0-9
print(random.randrange(1, 10, 2)) # 1,3,5,7,9 (步长)

# [a, b) 随机浮点数:
print(random.uniform(1.0, 5.0))  # 3.742...

# 随机选择:
items = ["苹果", "香蕉", "橘子", "葡萄"]
print(random.choice(items))      # 随机选一个

# 加权选择 (3.6+):
print(random.choices(items, weights=[5, 1, 1, 1], k=2))
# 苹果 5 倍概率,选 2 个

# 随机打乱:
deck = list(range(52))
random.shuffle(deck)             # 原地打乱
print(deck[:5])                  # [23, 7, 41, ...]

# 不重复采样:
print(random.sample(range(100), 5))  # [2, 45, 78, 13, 91]

# 设置种子 (可复现):
random.seed(42)
print(random.random())           # 0.639... (每次种子42都一样)

# 随机布尔值:
print(random.random() < 0.5)     # True (50%概率)
```


## random 模块: 分布与实用


```
// ========== 概率分布 ==========
import random

# 高斯分布 (正态分布):
# mu=均值, sigma=标准差
print(random.gauss(0, 1))        # 标准正态分布
print(random.normalvariate(0, 1))# 同上

# 指数分布:
print(random.expovariate(1.0))   # λ=1.0

# 三角分布:
print(random.triangular(0, 10, 5))  # min, max, mode

# 贝塔分布:
print(random.betavariate(2, 5))

# 伽马分布:
print(random.gammavariate(2, 2))

// ========== 实用示例 ==========
# 随机密码生成:
import string

def random_password(length=12):
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choice(chars) for _ in range(length))

print(random_password())          # 'aB3$kL9#xM2@'

# 随机颜色:
def random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

print(random_color())             # '#a3f0b2'

# 掷骰子:
def roll_dice(n=1, sides=6):
    return sum(random.randint(1, sides) for _ in range(n))

print(roll_dice(2))               # 7 (两枚六面骰子)
```


## statistics 模块


```
// ========== statistics 基础统计 ==========
import statistics

data = [2.5, 3.0, 4.5, 5.0, 3.5, 4.0, 6.0]

// ========== 集中趋势 ==========
print(statistics.mean(data))       # 4.071...  算术平均
print(statistics.median(data))     # 4.0       中位数
print(statistics.mode([1,1,2,3]))  # 1         众数 (最频值)

# 中位数变体:
print(statistics.median_low([1,2,3,4]))  # 2  低中位数
print(statistics.median_high([1,2,3,4])) # 3  高中位数

// ========== 离散程度 ==========
print(statistics.stdev(data))      # 1.202...  样本标准差 (n-1)
print(statistics.pstdev(data))     # 1.112...  总体标准差 (n)

print(statistics.variance(data))   # 1.445...  样本方差
print(statistics.pvariance(data))  # 1.237...  总体方差

// ========== 其他 ==========
# 协方差 (3.10+):
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
print(statistics.covariance(x, y)) # 5.0  协方差

# 相关系数 (3.10+):
print(statistics.correlation(x, y)) # 1.0  完全正相关

# 线性回归 (3.10+):
slope, intercept = statistics.linear_regression(x, y)
print(f"斜率={slope}, 截距={intercept}")  # 斜率=2.0, 截距=0.0

# 分位数 (3.8+):
print(statistics.quantiles(data, n=4))  # 四分位数
# [3.125, 4.0, 4.875] (Q1, Q2, Q3)
```


> **Note:** 💡 math/random/statistics 要点: (1) math: ceil/floor/sqrt/pow/log/三角函数; (2) random: random/randint/choice/shuffle/sample; (3) statistics: mean/median/stdev/variance; (4) random.seed(42) 固定随机种子确保可复现; (5) math.isclose() 安全比较浮点数。


## 练习


<!-- Converted from: 77_Python math与random与statistics模块.html -->
