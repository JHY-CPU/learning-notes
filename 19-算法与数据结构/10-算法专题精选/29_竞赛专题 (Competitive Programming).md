# 竞赛专题 (Competitive Programming)

## 一、概念定义与原理

### 1.1 主要竞赛

- **ICPC：** 三人组队，5小时，10+题
- **CCPC：** 国内版ICPC
- **Codeforces：** 个人赛，Rating系统
- **AtCoder：** 日本竞赛平台
- **LeetCode周赛：** 入门级竞赛

### 1.2 竞赛特点

- 时间压力大，需要快速编码
- 题目类型多样，需要广度+深度
- 对拍验证非常重要
- 常需要数学思维

---

## 二、竞赛必备技巧

### 2.1 代码模板

```cpp
// 万能头 + 快速IO
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define pii pair<int,int>
#define pb push_back
#define all(x) (x).begin(),(x).end()

void solve() {
    // 题目逻辑
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t; cin >> t;
    while (t--) solve();
}
```

### 2.2 对拍技巧

```bash
# 生成随机数据
python gen.py > input.txt
# 暴力解
./brute < input.txt > ans1.txt
# 正式解
./solution < input.txt > ans2.txt
# 比较
diff ans1.txt ans2.txt
```

### 2.3 常用 STL

```cpp
// 优先队列
priority_queue<int> maxpq; // 大顶堆
priority_queue<int, vector<int>, greater<int>> minpq; // 小顶堆

// 有序集合
set<int> s; // O(log n) 插入/删除/查找
multiset<int> ms; // 允许重复

// 自定义排序
sort(all(v), [](const auto& a, const auto& b) {
    return a.first < b.first || (a.first == b.first && a.second > b.second);
});

// 二分查找
lower_bound(all(v), x) - v.begin(); // 第一个 >= x
upper_bound(all(v), x) - v.begin(); // 第一个 > x

// 哈希
unordered_map<int,int> ump; // O(1) 期望

// 位运算
__builtin_popcount(x); // 1的个数
__builtin_ctz(x); // 尾部0的个数
__builtin_clz(x); // 前导0的个数
```

---

## 三、常见题型与策略

### 3.1 思维题

通常不需要复杂算法，关键在于找规律。

**策略：** 打表、枚举小数据找规律、数学证明。

### 3.2 贪心题

**策略：** 先猜想贪心策略，然后证明或对拍。

### 3.3 构造题

**策略：** 从小规模开始构造，逐步扩展。

### 3.4 交互题

**策略：** 二分搜索、信息论下界。

---

## 四、代码实现

### 4.1 快速IO模板 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;
const int MOD = 1e9 + 7;
const int INF = 1e18;

// 快速幂
ll qpow(ll a, ll b, ll m = MOD) {
    ll r = 1; a %= m;
    while (b) { if (b&1) r=r*a%m; a=a*a%m; b>>=1; }
    return r;
}

// 线性求逆元
const int MAXN = 2e6 + 5;
ll inv[MAXN];
void pre_inv() {
    inv[1] = 1;
    for (int i = 2; i < MAXN; i++) inv[i] = (MOD - MOD/i) * inv[MOD%i] % MOD;
}

// 组合数
ll fac[MAXN], ifac[MAXN];
void pre_comb() {
    fac[0] = 1;
    for (int i = 1; i < MAXN; i++) fac[i] = fac[i-1] * i % MOD;
    ifac[MAXN-1] = qpow(fac[MAXN-1], MOD-2);
    for (int i = MAXN-2; i >= 0; i--) ifac[i] = ifac[i+1] * (i+1) % MOD;
}
ll C(int n, int m) {
    if (m < 0 || m > n) return 0;
    return fac[n] % MOD * ifac[m] % MOD * ifac[n-m] % MOD;
}
```

### 4.2 Python 快速模板

```python
import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    # 题目逻辑

MOD = 10**9 + 7
def power(a, b, m=MOD):
    r = 1; a %= m
    while b:
        if b & 1: r = r * a % m
        a = a * a % m; b >>= 1
    return r

t = int(input())
for _ in range(t):
    solve()
```

### 4.3 常用数论模板

```cpp
// 质数判断
bool is_prime(ll n) {
    if (n < 2) return false;
    for (ll i = 2; i * i <= n; i++) if (n % i == 0) return false;
    return true;
}

// 因数分解
map<ll,int> factorize(ll n) {
    map<ll,int> res;
    for (ll i = 2; i * i <= n; i++) while (n % i == 0) { res[i]++; n /= i; }
    if (n > 1) res[n]++;
    return res;
}

// 扩展欧几里得
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (!b) { x = 1; y = 0; return a; }
    ll g = exgcd(b, a%b, y, x); y -= a/b*x; return g;
}
```

---

## 五、竞赛经验总结

### 5.1 时间分配

| 阶段 | 时间 | 任务 |
|------|------|------|
| 开场 | 15分钟 | 通读所有题，标注难度 |
| 中期 | 60% | AC 简单题和中等题 |
| 后期 | 30% | 攻坚难题 |
| 最后 | 10% | 检查、对拍 |

### 5.2 常见错误

1. **忘记清空全局变量：** 多组数据
2. **溢出：** `int` 乘法溢出，需用 `long long`
3. **读错题意：** 仔细读题
4. **边界条件：** $n=0, n=1$ 等特殊情况
5. **输出格式：** 注意空格、换行

### 5.3 提升建议

1. **坚持刷题：** Codeforces、AtCoder、洛谷
2. **总结模板：** 建立个人代码库
3. **学习题解：** 看高手的代码
4. **限时训练：** 模拟竞赛环境
5. **团队磨合：** ICPC 需要配合
